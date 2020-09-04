#!/usr/bin/env python3
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import options
from visual import plt as my_plt
from matplotlib.pyplot import get_cmap
import main_cl



## Parameter-values to compare
lamda_list = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000., 10000000000.,
              100000000000., 1000000000000.]
gamma_list = [1.]
c_list = [0.001, 0.01, 0.1, 1., 10. , 100., 1000., 10000., 100000., 1000000., 10000000., 100000000.]


# lamda_list = [1., 10.]
# gamma_list = [1.]
# c_list = [0.01, 0.1,]


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_MNIST': False}
    # Define input options
    parser = options.define_args(filename="main_cl", description='Select hyperparameters for EWC, online EWC and SI.')
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


def get_result(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile('{}/prec-{}.txt'.format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        main_cl.run(args)
    # -get average precision
    fileName = '{}/prec-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -return it
    return ave


if __name__ == '__main__':

    ## Load input-arguments & set default values
    args = handle_inputs()

    ## Add default arguments (will be different for different runs)
    args.ewc = False
    args.online = False
    args.si = False

    ## If needed, create plotting directory
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    ## Baselline
    BASE = get_result(args)

    ## EWC
    EWC = {}
    args.ewc = True
    for ewc_lambda in lamda_list:
        args.ewc_lambda=ewc_lambda
        EWC[ewc_lambda] = get_result(args)
    args.ewc = False

    ## Online EWC
    OEWC = {}
    args.ewc = True
    args.online = True
    for gamma in gamma_list:
        OEWC[gamma] = {}
        args.gamma = gamma
        for ewc_lambda in lamda_list:
            args.ewc_lambda = ewc_lambda
            OEWC[gamma][ewc_lambda] = get_result(args)
    args.ewc = False
    args.online = False

    ## SI
    SI = {}
    args.si = True
    for si_c in c_list:
        args.si_c = si_c
        SI[si_c] = get_result(args)
    args.si = False


    #-------------------------------------------------------------------------------------------------#

    #--------------------------------------------#
    #----- COLLECT DATA AND PRINT ON SCREEN -----#
    #--------------------------------------------#

    ext_c_list = [0] + c_list
    ext_lambda_list = [0] + lamda_list
    print("\n")


    ###---EWC + online EWC---###

    # -collect data
    ave_prec_ewc = [BASE] + [EWC[ewc_lambda] for ewc_lambda in lamda_list]
    ave_prec_per_lambda = [ave_prec_ewc]
    for gamma in gamma_list:
        ave_prec_temp = [BASE] + [OEWC[gamma][ewc_lambda] for ewc_lambda in lamda_list]
        ave_prec_per_lambda.append(ave_prec_temp)
    # -print on screen
    print("\n\nELASTIC WEIGHT CONSOLIDATION (EWC)")
    print(" param-list (lambda): {}".format(ext_lambda_list))
    print("  {}".format(ave_prec_ewc))
    print("--->  lambda = {}     --    {}".format(ext_lambda_list[np.argmax(ave_prec_ewc)], np.max(ave_prec_ewc)))
    if len(gamma_list) > 0:
        print("\n\nONLINE EWC")
        print(" param-list (lambda): {}".format(ext_lambda_list))
        curr_max = 0
        for gamma in gamma_list:
            ave_prec_temp = [BASE] + [OEWC[gamma][ewc_lambda] for ewc_lambda in lamda_list]
            print("  (gamma={}):   {}".format(gamma, ave_prec_temp))
            if np.max(ave_prec_temp) > curr_max:
                gamam_max = gamma
                lamda_max = ext_lambda_list[np.argmax(ave_prec_temp)]
                curr_max = np.max(ave_prec_temp)
        print("--->  gamma = {}  -  lambda = {}     --    {}".format(gamam_max, lamda_max, curr_max))


    ###---SI---###

    # -collect data
    ave_prec_si = [BASE] + [SI[c] for c in c_list]
    # -print on screen
    print("\n\nSYNAPTIC INTELLIGENCE (SI)")
    print(" param list (si_c): {}".format(ext_c_list))
    print("  {}".format(ave_prec_si))
    print("---> si_c = {}     --    {}".format(ext_c_list[np.argmax(ave_prec_si)], np.max(ave_prec_si)))
    print('\n')


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "hyperParams-{}-{}".format(args.experiment, args.tasks)
    scheme = "incremental task learning"
    title = "{}  -  {}".format(args.experiment, scheme)
    ylabel = "Average accuracy (after all tasks)"

    # calculate limits y-axes (to have equal for all graphs)
    full_list = [item for sublist in ave_prec_per_lambda for item in sublist] + ave_prec_si
    miny = np.min(full_list)
    maxy = np.max(full_list)
    marginy = 0.1*(maxy-miny)

    # open pdf
    pp = my_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []


    ###---EWC + online EWC---###
    # - select colors
    colors = ["darkgreen"]
    colors += get_cmap('Greens')(np.linspace(0.7, 0.3, len(gamma_list))).tolist()
    # - make plot (line plot - only average)
    figure = my_plt.plot_lines(ave_prec_per_lambda, x_axes=ext_lambda_list, ylabel=ylabel,
                               line_names=["EWC"] + ["Online EWC - gamma = {}".format(gamma) for gamma in gamma_list],
                               title=title, x_log=True, xlabel="EWC: lambda (log-scale)",
                               ylim=(miny-marginy, maxy+marginy),
                               with_dots=True, colors=colors, h_line=BASE, h_label="None")
    figure_list.append(figure)


    ###---SI---###
    figure = my_plt.plot_lines([ave_prec_si], x_axes=ext_c_list, ylabel=ylabel, line_names=["SI"],
                            colors=["yellowgreen"], title=title, x_log=True, xlabel="SI: c (log-scale)", with_dots=True,
                            ylim=(miny-marginy, maxy+marginy), h_line=BASE, h_label="None")
    figure_list.append(figure)


    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))
