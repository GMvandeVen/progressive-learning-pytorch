#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import torch
from torch import optim

# -custom-written libraries
import options
import utils
import define_models as define
from data.load import get_multitask_experiment
from eval import evaluate
from eval import callbacks as cb
from train import train_cl
from param_stamp import get_param_stamp
from models.cl.continual_learner import ContinualLearner
from models.cl.exemplars import ExemplarHandler
from visual import plt as visual_plt


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_MNIST': False}
    # Define input options
    parser = options.define_args(filename="main_cl", description='Compare approaches for task-incremental learning.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_allocation_options(parser, **kwargs)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    options.set_defaults(args, **kwargs)
    options.check_for_errors(args, **kwargs)
    return args


## Function for running one continual learning experiment
def run(args, verbose=False):

    # Create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if args.pdf and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    # If only want param-stamp, get it and exit
    if args.get_stamp:
        from param_stamp import get_param_stamp_from_args
        print(get_param_stamp_from_args(args=args))
        exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")

    # Report whether cuda is used
    if verbose:
        print("CUDA is {}used".format("" if cuda else "NOT(!!) "))

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)


    #-------------------------------------------------------------------------------------------------#

    #----------------#
    #----- DATA -----#
    #----------------#

    # Prepare data for chosen experiment
    if verbose:
        print("\nPreparing the data...")
    (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
        name=args.experiment, tasks=args.tasks, data_dir=args.d_dir,
        normalize=True if utils.checkattr(args, "normalize") else False,
        augment=True if utils.checkattr(args, "augment") else False,
        verbose=verbose, exception=True if args.seed<10 else False, only_test=(not args.train),
        max_samples=args.max_samples
    )


    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- MAIN MODEL -----#
    #----------------------#

    # Define main model (i.e., classifier, if requested with feedback connections)
    if verbose and utils.checkattr(args, "pre_convE")and (hasattr(args, "depth") and args.depth>0):
        print("\nDefining the model...")
    model = define.define_classifier(args=args, config=config, device=device)

    # Initialize / use pre-trained / freeze model-parameters
    # - initialize (pre-trained) parameters
    model = define.init_params(model, args)
    # - freeze weights of conv-layers?
    if utils.checkattr(args, "freeze_convE"):
        for param in model.convE.parameters():
            param.requires_grad = False

    # Define optimizer (only optimize parameters that "requires_grad")
    model.optim_list = [
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr},
    ]
    model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))


    #-------------------------------------------------------------------------------------------------#

    #----------------------------------#
    #----- CL-STRATEGY: EXEMPLARS -----#
    #----------------------------------#

    # Store in model whether, how many and in what way to store exemplars
    if isinstance(model, ExemplarHandler) and (args.use_exemplars or args.replay=="exemplars"):
        model.memory_budget = args.budget
        model.herding = args.herding
        model.norm_exemplars = args.herding


    #-------------------------------------------------------------------------------------------------#

    #----------------------------------------------------#
    #----- CL-STRATEGY: REGULARIZATION / ALLOCATION -----#
    #----------------------------------------------------#

    # Elastic Weight Consolidation (EWC)
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'ewc'):
        model.ewc_lambda = args.ewc_lambda if args.ewc else 0
        model.fisher_n = args.fisher_n
        model.online = utils.checkattr(args, 'online')
        if model.online:
            model.gamma = args.gamma

    # Synpatic Intelligence (SI)
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'si'):
        model.si_c = args.si_c if args.si else 0
        model.epsilon = args.epsilon

    # XdG: create for every task a "mask" for each hidden fully connected layer
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'xdg') and args.xdg_prop>0:
        model.define_XdGmask(gating_prop=args.xdg_prop, n_tasks=args.tasks)


    #-------------------------------------------------------------------------------------------------#

    #-------------------------------#
    #----- CL-STRATEGY: REPLAY -----#
    #-------------------------------#

    # Use distillation loss (i.e., soft targets) for replayed data? (and set temperature)
    if isinstance(model, ContinualLearner) and hasattr(args, 'replay') and not args.replay=="none":
        model.replay_targets = "soft" if args.distill else "hard"
        model.KD_temp = args.temp


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- REPORTING -----#
    #---------------------#

    # Get parameter-stamp (and print on screen)
    if verbose:
        print("\nParameter-stamp...")
    param_stamp, reinit_param_stamp = get_param_stamp(
        args, model.name, verbose=verbose,
        replay=True if (hasattr(args, 'replay') and not args.replay=="none") else False,
    )

    # Print some model-characteristics on the screen
    if verbose:
        # -main model
        utils.print_model_info(model, title="MAIN MODEL")

    # Prepare for keeping track of statistics required for metrics (also used for plotting in pdf)
    if args.pdf or args.metrics:
        # -define [metrics_dict] to keep track of performance during training for storing & for later plotting in pdf
        metrics_dict = evaluate.initiate_metrics_dict(n_tasks=args.tasks)
        # -evaluate randomly initiated model on all tasks & store accuracies in [metrics_dict] (for calculating metrics)
        if not args.use_exemplars:
            metrics_dict = evaluate.intial_accuracy(model, test_datasets, metrics_dict, no_task_mask=False,
                                                    classes_per_task=classes_per_task, test_size=None)
    else:
        metrics_dict = None

    # Prepare for plotting in visdom
    visdom = None
    if args.visdom:
        env_name = "{exp}-{tasks}".format(exp=args.experiment, tasks=args.tasks)
        replay_statement = "{mode}{b}".format(
            mode=args.replay,
            b="" if (args.batch_replay is None or args.batch_replay==args.batch) else "-br{}".format(args.batch_replay),
        ) if (hasattr(args, "replay") and not args.replay=="none") else "NR"
        graph_name = "{replay}{syn}{ewc}{xdg}".format(
            replay=replay_statement,
            syn="-si{}".format(args.si_c) if utils.checkattr(args, 'si') else "",
            ewc="-ewc{}{}".format(
                args.ewc_lambda,"-O{}".format(args.gamma) if utils.checkattr(args, "online") else ""
            ) if utils.checkattr(args, 'ewc') else "",
            xdg="" if (not utils.checkattr(args, 'xdg')) or args.xdg_prop==0 else "-XdG{}".format(args.xdg_prop),
        )
        visdom = {'env': env_name, 'graph': graph_name}


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#

    # Callbacks for reporting on and visualizing loss
    solver_loss_cbs = [
        cb._solver_loss_cb(log=args.loss_log, visdom=visdom, model=model, iters_per_task=args.iters, tasks=args.tasks,
                           replay=(hasattr(args, "replay") and not args.replay=="none"))
    ]

    # Callbacks for reporting and visualizing accuracy
    # -visdom (i.e., after each [prec_log]
    eval_cbs = [
        cb._eval_cb(log=args.prec_log, test_datasets=test_datasets, visdom=visdom,
                    iters_per_task=args.iters, test_size=args.prec_n, classes_per_task=classes_per_task,
                    with_exemplars=False)
    ] if (not args.use_exemplars) else [None]
    #--> during training on a task, evaluation cannot be with exemplars as those are only selected after training
    #    (instead, evaluation for visdom is only done after each task, by including callback-function into [metric_cbs])

    # Callbacks for calculating statists required for metrics
    # -pdf / reporting: summary plots (i.e, only after each task) (when using exemplars, also for visdom)
    metric_cbs = [
        cb._metric_cb(log=args.iters, test_datasets=test_datasets,
                      classes_per_task=classes_per_task, metrics_dict=metrics_dict,
                      iters_per_task=args.iters, with_exemplars=args.use_exemplars),
        cb._eval_cb(log=args.iters, test_datasets=test_datasets, visdom=visdom,
                    iters_per_task=args.iters, test_size=args.prec_n, classes_per_task=classes_per_task,
                    with_exemplars=True) if args.use_exemplars else None
    ]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    if args.train:
        if verbose:
            print("\nTraining...")
        # Train model
        train_cl(
            model, train_datasets, replay_mode=args.replay if hasattr(args, 'replay') else "none",
            classes_per_task=classes_per_task, iters=args.iters, args=args,
            batch_size=args.batch, batch_size_replay=args.batch_replay if hasattr(args, 'batch_replay') else None,
            eval_cbs=eval_cbs, loss_cbs=solver_loss_cbs, reinit=utils.checkattr(args, 'reinit'),
            only_last=utils.checkattr(args, 'only_last'), metric_cbs=metric_cbs, use_exemplars=args.use_exemplars,
        )
        # Save trained model(s), if requested
        if args.save:
            save_name = "mM-{}".format(param_stamp) if (
                not hasattr(args, 'full_stag') or args.full_stag == "none"
            ) else "{}-{}".format(model.name, args.full_stag)
            utils.save_checkpoint(model, args.m_dir, name=save_name, verbose=verbose)
    else:
        # Load previously trained model(s) (if goal is to only evaluate previously trained model)
        if verbose:
            print("\nLoading parameters of the previously trained models...")
        load_name = "mM-{}".format(param_stamp) if (
            not hasattr(args, 'full_ltag') or args.full_ltag == "none"
        ) else "{}-{}".format(model.name, args.full_ltag)
        utils.load_checkpoint(model, args.m_dir, name=load_name, verbose=verbose,
                              add_si_buffers=(isinstance(model, ContinualLearner) and utils.checkattr(args, 'si')))
        # Load previously created metrics-dict
        file_name = "{}/dict-{}".format(args.r_dir, param_stamp)
        metrics_dict = utils.load_object(file_name)

    #-------------------------------------------------------------------------------------------------#

    #-----------------------------------#
    #----- EVALUATION of CLASSIFIER-----#
    #-----------------------------------#

    if verbose:
        print("\n\nEVALUATION RESULTS:")

    # Evaluate precision of final model on full test-set
    precs = [evaluate.validate(
        model, test_datasets[i], verbose=False, test_size=None, task=i+1, with_exemplars=False,
        allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1)))
    ) for i in range(args.tasks)]
    average_precs = sum(precs) / args.tasks
    # -print on screen
    if verbose:
        print("\n Precision on test-set{}:".format(" (softmax classification)" if args.use_exemplars else ""))
        for i in range(args.tasks):
            print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
        print('=> Average precision over all {} tasks: {:.4f}\n'.format(args.tasks, average_precs))

    # -with exemplars
    if args.use_exemplars:
        precs = [evaluate.validate(
            model, test_datasets[i], verbose=False, test_size=None, task=i+1, with_exemplars=True,
            allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1)))
        ) for i in range(args.tasks)]
        average_precs_ex = sum(precs) / args.tasks
        # -print on screen
        if verbose:
            print(" Precision on test-set (classification using exemplars):")
            for i in range(args.tasks):
                print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
            print('=> Average precision over all {} tasks: {:.4f}\n'.format(args.tasks, average_precs_ex))

    # If requested, compute metrics
    if args.metrics:
        # Load accuracy matrix of "reinit"-experiment (i.e., each task's accuracy when only trained on that task)
        if not utils.checkattr(args, 'reinit'):
            file_name = "{}/dict-{}".format(args.r_dir, reinit_param_stamp)
            if not os.path.isfile("{}.pkl".format(file_name)):
                raise FileNotFoundError("Need to run the correct 'reinit' experiment (with --metrics) first!!")
            reinit_metrics_dict = utils.load_object(file_name)
        # Accuracy matrix
        R = pd.DataFrame(data=metrics_dict['acc per task'],
                         index=['after task {}'.format(i + 1) for i in range(args.tasks)])
        R = R[["task {}".format(task_id+1) for task_id in range(args.tasks)]]
        R.loc['at start'] = metrics_dict['initial acc per task'] if (not args.use_exemplars) else [
            'NA' for _ in range(args.tasks)
        ]
        if not utils.checkattr(args, 'reinit'):
            R.loc['only trained on itself'] = [
                reinit_metrics_dict['acc per task']['task {}'.format(task_id+1)][task_id] for task_id in range(args.tasks)
            ]
        R = R.reindex(['at start'] + ['after task {}'.format(i + 1) for i in range(args.tasks)]
                      + ['only trained on itself'])
        BWTs = [(R.loc['after task {}'.format(args.tasks), 'task {}'.format(i + 1)] - \
                 R.loc['after task {}'.format(i + 1), 'task {}'.format(i + 1)]) for i in range(args.tasks - 1)]
        FWTs = [0. if args.use_exemplars else (
            R.loc['after task {}'.format(i+1), 'task {}'.format(i + 2)] - R.loc['at start', 'task {}'.format(i+2)]
        ) for i in range(args.tasks-1)]
        forgetting = []
        for i in range(args.tasks - 1):
            forgetting.append(max(R.iloc[1:args.tasks, i]) - R.iloc[args.tasks, i])
        R.loc['FWT (per task)'] = ['NA'] + FWTs
        R.loc['BWT (per task)'] = BWTs + ['NA']
        R.loc['F (per task)'] = forgetting + ['NA']
        BWT = sum(BWTs) / (args.tasks - 1)
        F = sum(forgetting) / (args.tasks - 1)
        FWT = sum(FWTs) / (args.tasks - 1)
        metrics_dict['BWT'] = BWT
        metrics_dict['F'] = F
        metrics_dict['FWT'] = FWT
        # -Vogelstein et al's measures of transfer efficiency
        if not utils.checkattr(args, 'reinit'):
            TEs = [((1-R.loc['only trained on itself', 'task {}'.format(task_id+1)])
                    / (1-R.loc['after task {}'.format(args.tasks), 'task {}'.format(task_id+1)])) for task_id in range(args.tasks)]
            BTEs = [((1-R.loc['after task {}'.format(task_id+1), 'task {}'.format(task_id+1)])
                    / (1-R.loc['after task {}'.format(args.tasks), 'task {}'.format(task_id+1)])) for task_id in range(args.tasks)]
            FTEs = [((1-R.loc['only trained on itself', 'task {}'.format(task_id+1)])
                    / (1-R.loc['after task {}'.format(task_id+1), 'task {}'.format(task_id+1)])) for task_id in range(args.tasks)]
            # -TEs and BTEs after each task
            TEs_all = []
            BTEs_all = []
            for after_task_id in range(args.tasks):
                TEs_all.append([((1-R.loc['only trained on itself', 'task {}'.format(task_id+1)])
                                 / (1-R.loc['after task {}'.format(after_task_id+1), 'task {}'.format(task_id+1)])) for task_id in range(after_task_id+1)])
                BTEs_all.append([((1-R.loc['after task {}'.format(task_id+1), 'task {}'.format(task_id+1)])
                                 / (1-R.loc['after task {}'.format(after_task_id+1), 'task {}'.format(task_id+1)])) for task_id in range(after_task_id+1)])
            R.loc['TEs (per task, after all 10 tasks)'] = TEs
            for after_task_id in range(args.tasks):
                R.loc['TEs (per task, after {} tasks)'.format(after_task_id+1)] = TEs_all[after_task_id] + ['NA']*(args.tasks-after_task_id-1)
            R.loc['BTEs (per task, after all 10 tasks)'] = BTEs
            for after_task_id in range(args.tasks):
                R.loc['BTEs (per task, after {} tasks)'.format(after_task_id+1)] = BTEs_all[after_task_id] + ['NA']*(args.tasks-after_task_id-1)
            R.loc['FTEs (per task)'] = FTEs
            metrics_dict['R'] = R
        # -print on screen
        if verbose:
            print("Accuracy matrix")
            print(R)
            print("\nFWT = {:.4f}".format(FWT))
            print("BWT = {:.4f}".format(BWT))
            print("  F = {:.4f}\n\n".format(F))


    #-------------------------------------------------------------------------------------------------#

    #------------------#
    #----- OUTPUT -----#
    #------------------#

    # Average precision on full test set
    output_file = open("{}/prec-{}.txt".format(args.r_dir, param_stamp), 'w')
    output_file.write('{}\n'.format(average_precs_ex if args.use_exemplars else average_precs))
    output_file.close()
    # -metrics-dict
    if args.metrics:
        file_name = "{}/dict-{}".format(args.r_dir, param_stamp)
        utils.save_object(metrics_dict, file_name)


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # If requested, generate pdf
    if args.pdf:
        # -open pdf
        plot_name = "{}/{}.pdf".format(args.p_dir, param_stamp)
        pp = evaluate.visual.plt.open_pdf(plot_name)

        # -plot TEs
        if not utils.checkattr(args, 'reinit'):
            BTEs = []
            for task_id in range(args.tasks):
                BTEs.append([R.loc['BTEs (per task, after {} tasks)'.format(after_task_id+1), 'task {}'.format(task_id+1)] for after_task_id in range(task_id, args.tasks)])
            figure = visual_plt.plot_TEs([FTEs], [BTEs], [TEs], ["test"])
            pp.savefig(figure)

        # -show metrics reflecting progression during training
        if args.train and (not utils.checkattr(args, 'only_last')):
            # -create list to store all figures to be plotted.
            figure_list = []
            # -generate all figures (and store them in [figure_list])
            key = "acc per task"
            plot_list = []
            for i in range(args.tasks):
                plot_list.append(metrics_dict[key]["task {}".format(i + 1)])
            figure = visual_plt.plot_lines(
                plot_list, x_axes=metrics_dict["x_task"],
                line_names=['task {}'.format(i + 1) for i in range(args.tasks)]
            )
            figure_list.append(figure)
            figure = visual_plt.plot_lines(
                [metrics_dict["average"]], x_axes=metrics_dict["x_task"],
                line_names=['average all tasks so far']
            )
            figure_list.append(figure)
            # -add figures to pdf
            for figure in figure_list:
                pp.savefig(figure)

        # -close pdf
        pp.close()

        # -print name of generated plot on screen
        if verbose:
            print("\nGenerated plot: {}\n".format(plot_name))



if __name__ == '__main__':
    args = handle_inputs()
    run(args, verbose=True)