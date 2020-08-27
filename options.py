import argparse
from utils import checkattr


##-------------------------------------------------------------------------------------------------------------------##

####################
## Define options ##
####################

def define_args(filename, description):
    parser = argparse.ArgumentParser('./{}.py'.format(filename), description=description)
    return parser


def add_general_options(parser, single_task=False, only_MNIST=True, **kwargs):
    parser.add_argument('--no-save', action='store_false', dest='save', help="don't save trained models")
    if not only_MNIST:
        parser.add_argument('--convE-stag', type=str, metavar='STAG', default='none',help="tag for saving convE-layers")
    parser.add_argument('--full-stag', type=str, metavar='STAG', default='none', help="tag for saving full model")
    parser.add_argument('--full-ltag', type=str, metavar='LTAG', default='none', help="tag for loading full model")
    parser.add_argument('--test', action='store_false', dest='train', help='evaluate previously saved model')
    if not single_task:
        parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
    parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
    parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
    parser.add_argument('--data-dir', type=str, default='./store/datasets', dest='d_dir', help="default: %(default)s")
    parser.add_argument('--model-dir', type=str, default='./store/models', dest='m_dir', help="default: %(default)s")
    parser.add_argument('--plot-dir', type=str, default='./store/plots', dest='p_dir', help="default: %(default)s")
    if not single_task:
        parser.add_argument('--results-dir', type=str, default='./store/results', dest='r_dir',
                            help="default: %(default)s")
    return parser


def add_eval_options(parser, single_task=False, **kwargs):
    # evaluation parameters
    eval = parser.add_argument_group('Evaluation Parameters')
    eval.add_argument('--metrics', action='store_true', help="calculate additional metrics (e.g., BWT, forgetting)")
    eval.add_argument('--pdf', action='store_true', help="generate pdf with plots for individual experiment(s)")
    eval.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
    eval.add_argument('--log-per-task', action='store_true', help="set all visdom-logs to [iters]")
    eval.add_argument('--loss-log', type=int, default=500, metavar="N", help="# iters after which to plot loss")
    eval.add_argument('--prec-log', type=int, default=None if single_task else 500, metavar="N",
                      help="# iters after which to plot precision")
    eval.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating accuracy (visdom-plots)")
    return parser


def add_task_options(parser, only_MNIST=False, single_task=False, **kwargs):
    # expirimental task parameters
    task_params = parser.add_argument_group('Task Parameters')
    if single_task:
        task_choices = ['CIFAR10', 'CIFAR100', 'MNIST', 'MNIST28']
        task_default = 'CIFAR10'
    else:
        MNIST_tasks = ['splitMNIST', 'permMNIST']
        image_tasks = ['CIFAR100']
        task_choices = MNIST_tasks if only_MNIST else MNIST_tasks+image_tasks
        task_default = 'splitMNIST' if only_MNIST else 'CIFAR100'
    task_params.add_argument('--experiment', type=str, default=task_default, choices=task_choices)
    if not single_task:
        task_params.add_argument('--tasks', type=int, help='number of tasks')
    if not only_MNIST:
        task_params.add_argument('--augment', action='store_true',
                                 help="augment training data (random crop & horizontal flip)")
        task_params.add_argument('--no-norm', action='store_false', dest='normalize',
                                 help="don't normalize images (only for CIFAR)")
    task_params.add_argument('--only-last', action='store_true', help="only train on last task / episode")
    return parser


def add_model_options(parser, only_MNIST=False, single_task=False, **kwargs):
    # model architecture parameters
    model = parser.add_argument_group('Parameters Main Model')
    if not only_MNIST:
        # -conv-layers
        model.add_argument('--conv-type', type=str, default="standard", choices=["standard", "resNet"])
        model.add_argument('--n-blocks', type=int, default=2, help="# blocks per conv-layer (only for 'resNet')")
        model.add_argument('--depth', type=int, default=5 if single_task else None,
                           help="# of convolutional layers (0 = only fc-layers)")
        model.add_argument('--reducing-layers', type=int, dest='rl',help="# of layers with stride (=image-size halved)")
        model.add_argument('--channels', type=int, default=16, help="# of channels 1st conv-layer (doubled every 'rl')")
        model.add_argument('--conv-bn', type=str, default="yes", help="use batch-norm in the conv-layers (yes|no)")
        model.add_argument('--conv-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
        model.add_argument('--global-pooling', action='store_true', dest='gp', help="ave global pool after conv-layers")
    # -fully-connected-layers
    model.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
    model.add_argument('--fc-units', type=int, default=2000 if single_task else None, metavar="N",
                       help="# of units in first fc-layers")
    model.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
    model.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
    model.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu", "none"])
    model.add_argument('--h-dim', type=int, metavar="N", help='# of hidden units final layer (default: fc-units)')
    # NOTE: number of units per fc-layer linearly declinces from [fc_units] to [h_dim].
    return parser


def add_train_options(parser, only_MNIST=False, single_task=False, **kwargs):
    # training hyperparameters / initialization
    train_params = parser.add_argument_group('Training Parameters')
    if single_task:
        iter_epochs = train_params.add_mutually_exclusive_group(required=False)
        iter_epochs.add_argument('--epochs', type=int, default=10, metavar='N',
                                 help='max # of epochs (default: %(default)d)')
        iter_epochs.add_argument('--iters', type=int, metavar='N', help='max # of iterations (replaces "--epochs")')
    else:
        train_params.add_argument('--iters', type=int, help="# batches to optimize main model")
    train_params.add_argument('--lr', type=float, default=0.0001 if single_task else None, help="learning rate")
    train_params.add_argument('--batch', type=int, default=256 if single_task else None, help="batch-size")
    train_params.add_argument('--init-weight', type=str, default='standard', choices=['standard', 'xavier'])
    train_params.add_argument('--init-bias', type=str, default='standard', choices=['standard', 'constant'])
    train_params.add_argument('--reinit', action='store_true', help='reinitialize networks before each new task')
    if not only_MNIST:
        train_params.add_argument('--pre-convE', action='store_true', help="use pretrained convE-layers")
        train_params.add_argument('--convE-ltag', type=str, metavar='LTAG', default='none',
                                  help="tag for loading convE-layers")
        train_params.add_argument('--freeze-convE', action='store_true', help="freeze parameters of convE-layers")
    return parser


def add_replay_options(parser, **kwargs):
    # replay parameters
    replay = parser.add_argument_group('Replay Parameters')
    replay_choices = ['offline', 'none', 'current', 'exact', 'exemplars']
    replay.add_argument('--replay', type=str, default='none', choices=replay_choices)
    replay.add_argument('--distill', action='store_true', help="use distillation for replay")
    replay.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    replay.add_argument('--agem', action='store_true', help="use gradient of replay as inequality constraint")
    replay.add_argument('--batch-replay', type=int, metavar='N', help="batch-size for replay (default: batch)")
    # exemplar parameters
    icarl_params = parser.add_argument_group('Exemplar Parameters')
    icarl_params.add_argument('--use-exemplars', action='store_true', help="use exemplars for classification")
    icarl_params.add_argument('--budget', type=int, default=1000, help="how many exemplars can be stored?")
    icarl_params.add_argument('--herding', action='store_true', help="use herding to select exemplars (else random)")
    return parser


def add_allocation_options(parser,  **kwargs):
    cl = parser.add_argument_group('Memory Allocation Parameters')
    cl.add_argument('--ewc', action='store_true', help="use 'EWC' (Kirkpatrick et al, 2017)")
    cl.add_argument('--lambda', type=float, dest="ewc_lambda",help="--> EWC: regularisation strength")
    cl.add_argument('--online', action='store_true', help="--> EWC: perform 'online EWC'")
    cl.add_argument('--gamma', type=float, help="--> EWC: forgetting coefficient (for 'online EWC')")
    cl.add_argument('--fisher-n', type=int, default=1000, help="--> EWC: sample size estimating Fisher Information")
    cl.add_argument('--si', action='store_true', help="use 'Synaptic Intelligence' (Zenke, Poole et al, 2017)")
    cl.add_argument('--c', type=float, dest="si_c", help="-->  SI: regularisation strength")
    cl.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="-->  SI: dampening parameter")
    cl.add_argument('--xdg', action='store_true', help="use 'Context-dependent Gating' (Masse et al, 2018)")
    cl.add_argument('--xdg-prop', type=float, dest='xdg_prop', help="--> XdG: prop neurons per layer to gate")
    return  parser


##-------------------------------------------------------------------------------------------------------------------##

############################
## Check / modify options ##
############################

def set_defaults(args, single_task=False, **kwargs):
    # -if [iCaRL] is selected, select all accompanying options
    if hasattr(args, "icarl") and args.icarl:
        args.use_exemplars = True
        args.add_exemplars = True
        args.bce = True
        args.bce_distill = True
    # -set default-values for certain arguments based on chosen experiment
    args.normalize = args.normalize if args.experiment in ('CIFAR10', 'CIFAR100') else False
    args.augment = args.augment if args.experiment in ('CIFAR10', 'CIFAR100') else False
    if hasattr(args, "depth"):
        args.depth = (5 if args.experiment in ('CIFAR10', 'CIFAR100') else 0) if args.depth is None else args.depth
    if not single_task:
        args.tasks= (
            5 if args.experiment=='splitMNIST' else (10 if args.experiment=="CIFAR100" else 100)
        ) if args.tasks is None else args.tasks
        args.iters = 2000 if args.iters is None else args.iters
        args.lr = (0.001 if args.experiment=='splitMNIST' else 0.0001) if args.lr is None else args.lr
        args.batch = (128 if args.experiment=='splitMNIST' else 256) if args.batch is None else args.batch
        args.fc_units = (400 if args.experiment=='splitMNIST' else 2000) if args.fc_units is None else args.fc_units
    # -set hyper-parameter values (typically found by grid-search) based on chosen experiment
    if not single_task:
        if args.experiment=='splitMNIST':
            args.xdg_prop = 0.9 if args.xdg_prop is None else args.xdg_prop
            args.si_c = 10. if args.si_c is None else args.si_c
            args.ewc_lambda = 1000000000. if args.ewc_lambda is None else args.ewc_lambda
            args.gamma = 1. if args.gamma is None else args.gamma
        elif args.experiment=='CIFAR100':
            args.xdg_prop = 0.7 if args.xdg_prop is None else args.xdg_prop
            args.si_c = 100. if args.si_c is None else args.si_c
            args.ewc_lambda = 1000. if args.ewc_lambda is None else args.ewc_lambda
            args.gamma = 1 if args.gamma is None else args.gamma
        elif args.experiment=='permMNIST':
            args.si_c = 10. if args.si_c is None else args.si_c
            args.ewc_lambda = 1. if args.ewc_lambda is None else args.ewc_lambda
            args.gamma = 1. if args.gamma is None else args.gamma
    # -for other unselected options, set default values (not specific to chosen experiment)
    args.h_dim = args.fc_units if args.h_dim is None else args.h_dim
    if hasattr(args, "rl"):
        args.rl = args.depth-1 if args.rl is None else args.rl
    args.xdg_prop = 0. if args.xdg_prop is None else args.xdg_prop
    # -if [log_per_task], reset all logs
    if checkattr(args, 'log_per_task'):
        args.prec_log = args.iters
        args.loss_log = args.iters
        args.sample_log = args.iters
    return args


def check_for_errors(args, single_task=False, **kwargs):
    # -errors in chosen options
    if not single_task:
        # -if XdG is selected together with replay of any kind, give error
        if checkattr(args, 'xdg') and args.xdg_prop>0 and (not args.replay=="none"):
            raise NotImplementedError("XdG is not supported with '{}' replay.".format(args.replay))
            #--> problem is that applying different task-masks interferes with gradient calculation
            #    (should be possible to overcome by calculating each gradient before applying next mask)
        # -if 'only_last' is selected with replay, EWC or SI, give error
        if checkattr(args, 'only_last') and (not args.replay=="none"):
            raise NotImplementedError("Option 'only_last' is not supported with '{}' replay.".format(args.replay))
        if checkattr(args, 'only_last') and (checkattr(args, 'ewc') and args.ewc_lambda>0):
            raise NotImplementedError("Option 'only_last' is not supported with EWC.")
        if checkattr(args, 'only_last') and (checkattr(args, 'si') and args.si_c>0):
            raise NotImplementedError("Option 'only_last' is not supported with SI.")