#!/usr/bin/env python3
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import visual.plt as visual_plt
import main_cl
import options
import utils


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_MNIST': False}
    # Define input options
    parser = options.define_args(filename="main_cl", description='Compare & combine continual learning approaches.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_allocation_options(parser, **kwargs)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    parser.add_argument('--n-seeds', type=int, default=1)
    parser.add_argument('--exact', action='store_true', help="use 'exact' replay, instead of 'offline' replay")
    args = parser.parse_args()
    options.set_defaults(args, **kwargs)
    options.check_for_errors(args, **kwargs)
    return args


def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run; if not do so
    if os.path.isfile('{}/dict-{}.pkl'.format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        args.metrics = True
        main_cl.run(args)
    # -get average precision
    file_name = '{}/prec-{}.txt'.format(args.r_dir, param_stamp)
    file = open(file_name)
    ave = float(file.readline())
    file.close()
    # -get metrics-dict
    file_name = '{}/dict-{}'.format(args.r_dir, param_stamp)
    metrics_dict = utils.load_object(file_name)
    # -print average precision on screen
    print("--> average precision: {}".format(ave))
    # -return average precision & metrics-dict
    return (ave, metrics_dict)


def collect_all(method_dict, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_results(args)
    # -return updated dictionary with results
    return method_dict


def calc_mean_bte(btes, task_num=10, reps=6):
    mean_bte = [[] for i in range(task_num)]

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(btes[i][j])

        tmp = tmp / reps
        mean_bte[j].extend(tmp)

    return mean_bte


def calc_mean_te(tes):
    fte = np.asarray(tes)
    return list(np.mean(np.asarray(fte), axis=0))



if __name__ == '__main__':

    ## Load input-arguments
    args = handle_inputs()
    # -create results-directory if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    # -create plots-directory if needed
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    ## Add non-optional input argument that will be the same for all runs
    args.log_per_task = True

    ## Add input arguments that will be different for different runs
    args.distill = False
    args.agem = False
    args.ewc = False
    args.online = False
    args.si = False
    args.xdg = False
    args.add_exemplars = False
    args.bce_distill= False
    args.icarl = False
    # args.seed could of course also vary!

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))

    ## WITH PRETRAINED CONV-LAYERS
    args.pre_convE = True
    args.convE_ltag = "s100N"

    ## Only train on each individual task
    args.reinit = True
    REINIT_P = {}
    REINIT_P = collect_all(REINIT_P, seed_list, args,
                           name="Only train on each individual task (using 'reinit' - pretrained)")
    args.reinit = False

    ## Replay All
    args.replay = "exact" if args.exact else "offline"
    OFF_P = {}
    OFF_P = collect_all(OFF_P, seed_list, args, name="Replay All (pre-trained conv-layers)")
    args.replay = "none"


    ## WITHOUT PRETRAINED CONV-LAYERS
    args.pre_convE = False
    args.convE_ltag = None

    ## Only train on each individual task
    args.reinit = True
    REINIT = {}
    REINIT = collect_all(REINIT, seed_list, args, name="Only train on each individual task (using 'reinit')")
    args.reinit = False

    ## Replay All
    args.replay = "exact" if args.exact else "offline"
    OFF = {}
    OFF = collect_all(OFF, seed_list, args, name="Replay All")
    args.replay = "none"



    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    ave_prec = {}
    metric_dict = {}
    prec = {}

    ## For each seed, create list with average precisions
    for seed in seed_list:
        i = 0
        ave_prec[seed] = [OFF[seed][i], OFF_P[seed][i]]

        i = 1
        metric_dict[seed] = [OFF[seed][i], OFF_P[seed][i]]

        key = "average"
        prec[seed] = [OFF[seed][i][key] , OFF_P[seed][i][key]]

    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summary-{}-{}".format(args.experiment, args.tasks)
    scheme = "task-incremental learning"
    title = "{}  -  {}".format(args.experiment, scheme)

    # select names / colors / ids
    names = ["Replay All, no pre-training", "Replay All, pre-trained conv-layers"]
    colors = ["black", "red"]
    ids = [0,1]

    # open pdf
    pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # bar-plot
    means = [np.mean([ave_prec[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems = [np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
        cis = [1.96*np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    # figure = visual_plt.plot_bar(means, names=names, colors=colors, ylabel="average precision (after all tasks)",
    #                              title=title, yerr=cis if len(seed_list)>1 else None, ylim=(0,1))
    # figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"-"*60)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:18s} {:.2f}  (+/- {:.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:18s} {:.2f}".format(name, 100*means[i]))
    print("#"*60)

    # # line-plot
    # x_axes = OFF[args.seed][1]["x_task"]
    # ave_lines = []
    # sem_lines = []
    # for id in ids:
    #     new_ave_line = []
    #     new_sem_line = []
    #     for line_id in range(len(prec[args.seed][id])):
    #         all_entries = [prec[seed][id][line_id] for seed in seed_list]
    #         new_ave_line.append(np.mean(all_entries))
    #         if len(seed_list) > 1:
    #             new_sem_line.append(1.96*np.sqrt(np.var(all_entries)/(len(all_entries)-1)))
    #     ave_lines.append(new_ave_line)
    #     sem_lines.append(new_sem_line)
    # figure = visual_plt.plot_lines(ave_lines, x_axes=x_axes, line_names=names, colors=colors, title=title,
    #                                xlabel="# of tasks", ylabel="Average precision (on tasks seen so far)",
    #                                list_with_errors=sem_lines if len(seed_list)>1 else None)
    # figure_list.append(figure)


    # Plot Transfer Efficiency
    BTEs = []
    FTEs = []
    TEs = []
    for id in ids:
        BTEs_this_alg = []
        FTEs_this_alg = []
        TEs_this_alg = []
        for seed in seed_list:
            R = metric_dict[seed][id]['R']
            TEs_this_alg_this_seed = R.loc['TEs (per task, after all 10 tasks)']
            FTEs_this_alg_this_seed = R.loc['FTEs (per task)']
            BTEs_this_alg_this_seed = []
            for task_id in range(args.tasks):
                BTEs_this_alg_this_seed.append(
                    [R.loc['BTEs (per task, after {} tasks)'.format(after_task_id + 1), 'task {}'.format(task_id + 1)] for
                     after_task_id in range(task_id, args.tasks)]
                )
                BTEs_this_alg.append(BTEs_this_alg_this_seed)
                FTEs_this_alg.append(FTEs_this_alg_this_seed)
                TEs_this_alg.append(TEs_this_alg_this_seed)
        BTEs.append(calc_mean_bte(BTEs_this_alg, task_num=args.tasks, reps=len(seed_list)))
        FTEs.append(calc_mean_te(FTEs_this_alg))
        TEs.append(calc_mean_te(TEs_this_alg))
    figure = visual_plt.plot_TEs(FTEs, BTEs, TEs, names, task_num=args.tasks)
    figure_list.append(figure)


    # add all figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))