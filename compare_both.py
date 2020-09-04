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
    parser = options.define_args(filename="main_cl",
                                 description='Compare CL approaches in terms of transfer efficiency.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_allocation_options(parser, **kwargs)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    parser.add_argument('--n-seeds', type=int, default=1)
    parser.add_argument('--o-lambda', metavar="LAMBDA", type=float, help="--> Online EWC: regularisation strength")
    parser.add_argument('--c-500', metavar="C", type=float, help="--> SI: reg strength with 500 training samples")
    parser.add_argument('--lambda-500', metavar="LAMBDA", type=float,
                        help="--> EWC: reg strength with 500 training samples")
    parser.add_argument('--o-lambda-500', metavar="LAMBDA", type=float,
                        help="--> Online EWC: reg strength with 500 training samples")
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

    ###----"Re-init"----###
    args.reinit = True
    REINIT = {}
    REINIT = collect_all(REINIT, seed_list, args, name="Only train on each individual task (using 'reinit')")
    args.max_samples = 50
    args.iters = 500
    REINITp = {}
    REINITp = collect_all(REINITp, seed_list, args, name="Only train on each individual task (using 'reinit' - 500 samples)")
    args.max_samples = None
    args.iters = 5000
    args.reinit = False

    ## None
    args.replay = "none"
    NONE = {}
    NONE = collect_all(NONE, seed_list, args, name="None")
    args.max_samples = 50
    args.iters = 500
    NONEp = {}
    NONEp = collect_all(NONEp, seed_list, args, name="None - 500 samples")
    args.max_samples = None
    args.iters = 5000

    ## Offline
    args.replay = "offline"
    OFF = {}
    OFF = collect_all(OFF, seed_list, args, name="Full replay (increasing amount of replay with each new task)")
    args.max_samples = 50
    args.iters = 500
    OFFp = {}
    OFFp = collect_all(OFFp, seed_list, args, name="Full replay (increasing amount of replay with each new task - 500 samples)")
    args.max_samples = None
    args.iters = 5000
    args.replay = "none"

    ## Exact replay
    args.replay = "exact"
    EXACT = {}
    EXACT = collect_all(EXACT, seed_list, args, name="Exact replay (fixed amount of total replay)")
    args.max_samples = 50
    args.iters = 500
    EXACTp = {}
    EXACTp = collect_all(EXACTp, seed_list, args, name="Exact replay (fixed amount of total replay - 500 samples)")
    args.max_samples = None
    args.iters = 5000
    args.replay = "none"

    ## EWC
    args.ewc = True
    EWC = {}
    EWC = collect_all(EWC, seed_list, args, name="EWC")
    args.max_samples = 50
    args.iters = 500
    args.ewc_lambda = args.lambda_500 if args.lambda_500 is not None else args.ewc_lambda
    EWCp = {}
    EWCp = collect_all(EWCp, seed_list, args, name="EWC - 500 samples")
    args.max_samples = None
    args.iters = 5000

    ## online EWC
    args.online = True
    args.ewc = True
    args.ewc_lambda = args.o_lambda
    OEWC = {}
    OEWC = collect_all(OEWC, seed_list, args, name="Online EWC")
    args.max_samples = 50
    args.iters = 500
    args.ewc_lambda = args.o_lambda_500 if args.o_lambda_500 is not None else args.ewc_lambda
    OEWCp = {}
    OEWCp = collect_all(OEWCp, seed_list, args, name="Online EWC - 500 samples")
    args.max_samples = None
    args.iters = 5000
    args.ewc = False
    args.online = False

    ## SI
    args.si = True
    SI = {}
    SI = collect_all(SI, seed_list, args, name="SI")
    args.max_samples = 50
    args.iters = 500
    args.si_c = args.c_500 if args.c_500 is not None else args.si_c
    SIp = {}
    SIp = collect_all(SIp, seed_list, args, name="SI - 500 samples")
    args.max_samples = None
    args.iters = 5000
    args.si = False

    ## LwF
    args.replay = "current"
    args.distill = True
    LWF = {}
    LWF = collect_all(LWF, seed_list, args, name="LwF")
    args.max_samples = 50
    args.iters = 500
    LWFp = {}
    LWFp = collect_all(LWFp, seed_list, args, name="LwF - 500 samples")
    args.max_samples = None
    args.iters = 5000


    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    ave_prec = {}
    metric_dict = {}
    ave_prec_p = {}
    metric_dict_p = {}

    ## For each seed, create list with average precisions
    for seed in seed_list:
        i = 0
        ave_prec[seed] = [OFF[seed][i], NONE[seed][i], EWC[seed][i], OEWC[seed][i], SI[seed][i],
                          LWF[seed][i], EXACT[seed][i]]

        i = 1
        metric_dict[seed] = [OFF[seed][i], NONE[seed][i], EWC[seed][i], OEWC[seed][i], SI[seed][i],
                             LWF[seed][i], EXACT[seed][i]]

        i = 0
        ave_prec_p[seed] = [OFFp[seed][i], NONEp[seed][i], EWCp[seed][i], OEWCp[seed][i], SIp[seed][i],
                            LWFp[seed][i], EXACTp[seed][i]]

        i = 1
        metric_dict_p[seed] = [OFFp[seed][i], NONEp[seed][i], EWCp[seed][i], OEWCp[seed][i], SIp[seed][i],
                               LWFp[seed][i], EXACTp[seed][i]]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summary-{}-{}".format(args.experiment, args.tasks)
    title = "{}".format(args.experiment)

    # select names / colors / ids
    names = ["Replay (increasing amount)", "Replay (fixed amount)", "EWC", "Online EWC", "SI", "LwF", "None"]
    short_names = ["Replay (increasing)", "Replay (fixed)", "EWC", "Online EWC", "SI", "LwF", "None"]
    colors = ["darkred", "red", "dodgerblue", "darkblue", "green", "goldenrod", "grey"]
    ids = [0,6,2,3,4,5,1]

    # open pdf
    pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # print average accuracy
    # -not pretrained
    means = [np.mean([ave_prec[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems = [np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    print("\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"-"*60)
    for i,name in enumerate(short_names):
        if len(seed_list) > 1:
            print("{:22s} {:.2f}  (+/- {:.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:22s} {:.2f}".format(name, 100*means[i]))
    print("#"*60)
    # -pretrained
    means = [np.mean([ave_prec_p[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems = [np.sqrt(np.var([ave_prec_p[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    print("\n\n"+"#"*60+"\nSUMMARY RESULTS (pre-trained): {}\n".format(title)+"-"*60)
    for i,name in enumerate(short_names):
        if len(seed_list) > 1:
            print("{:22s} {:.2f}  (+/- {:.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:22s} {:.2f}".format(name, 100*means[i]))
    print("#"*60)

    # plot Transfer Efficiency
    # -collect not pretrained
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
    # -collect pretrained
    BTEsp = []
    FTEsp = []
    TEsp = []
    for id in ids:
        BTEs_this_alg = []
        FTEs_this_alg = []
        TEs_this_alg = []
        for seed in seed_list:
            R = metric_dict_p[seed][id]['R']
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
        BTEsp.append(calc_mean_bte(BTEs_this_alg, task_num=args.tasks, reps=len(seed_list)))
        FTEsp.append(calc_mean_te(FTEs_this_alg))
        TEsp.append(calc_mean_te(TEs_this_alg))
    # -make plot
    figure = visual_plt.plot_TEs_twice(FTEsp, BTEsp, TEsp, FTEs, BTEs, TEs, names,
                                       top_title="500 training samples per task",
                                       bottom_title="5000 training samples per task",
                                       short_names=short_names, task_num=args.tasks, y_lim=(0.58, 1.32),
                                       colors=colors)
    figure_list.append(figure)


    # add all figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))