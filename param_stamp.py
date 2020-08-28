from data.load import get_multitask_experiment
from utils import checkattr


def get_param_stamp_from_args(args):
    '''To get param-stamp a bit quicker.'''
    from define_models import define_classifier

    # -get configurations of experiment
    config = get_multitask_experiment(
        name=args.experiment, tasks=args.tasks, data_dir=args.d_dir, only_config=True,
        normalize=args.normalize if hasattr(args, "normalize") else False, verbose=False,
    )

    # -get model architectures
    model = define_classifier(args=args, config=config, device='cpu')
    train_gen = (hasattr(args, 'replay') and args.replay=="generative" and not checkattr(args, 'feedback'))

    # -extract and return param-stamp
    model_name = model.name
    param_stamp, _ = get_param_stamp(args, model_name, replay=(hasattr(args, "replay") and not args.replay=="none"),
                                    verbose=False)
    return param_stamp



def get_param_stamp(args, model_name, verbose=True, replay=False):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for task
    multi_n_stamp = "{n}{of}".format(
        n=args.tasks, of="OL" if checkattr(args, 'only_last') else ""
    ) if hasattr(args, "tasks") else ""
    task_stamp = "{exp}{norm}{aug}{multi_n}{max_n}".format(
        exp=args.experiment, norm="-N" if hasattr(args, 'normalize') and args.normalize else "",
        aug="+" if hasattr(args, "augment") and args.augment else "", multi_n=multi_n_stamp,
        max_n="" if (not args.experiment=="CIFAR100") or args.max_samples is None else "-max{}".format(args.max_samples)
    )
    if verbose:
        print(" --> task:          "+task_stamp)

    # -for model
    model_stamp = model_name
    if verbose:
        print(" --> model:         "+model_stamp)

    # -for hyper-parameters
    pre_conv = ""
    if checkattr(args, "pre_convE") and (hasattr(args, 'depth') and args.depth>0):
        ltag = "" if not hasattr(args, "convE_ltag") or args.convE_ltag=="none" else "-{}".format(args.convE_ltag)
        pre_conv = "-pCvE{}".format(ltag)
    freeze_conv = "-fCvE" if (checkattr(args, "freeze_convE") and hasattr(args, 'depth') and args.depth>0) else ""
    hyper_stamp = "{i_e}{num}-lr{lr}-b{bsz}{pretr}{freeze}{reinit}".format(
        i_e="e" if args.iters is None else "i", num=args.epochs if args.iters is None else args.iters, lr=args.lr,
        bsz=args.batch, pretr=pre_conv, freeze=freeze_conv, reinit="-R" if checkattr(args, 'reinit') else ""
    )
    if verbose:
        print(" --> hyper-params:  " + hyper_stamp)

    # -for EWC / SI
    if (checkattr(args, 'ewc') and args.ewc_lambda>0) or (checkattr(args, 'si') and args.si_c>0):
        ewc_stamp = "EWC{l}-{fi}{o}".format(
            l=args.ewc_lambda, fi="{}".format("N" if args.fisher_n is None else args.fisher_n),
            o="-O{}".format(args.gamma) if checkattr(args, 'online') else "",
        ) if (checkattr(args, 'ewc') and args.ewc_lambda>0) else ""
        si_stamp = "SI{c}-{eps}".format(c=args.si_c, eps=args.epsilon) if (checkattr(args,'si') and args.si_c>0) else ""
        both = "--" if (checkattr(args,'ewc') and args.ewc_lambda>0) and (checkattr(args,'si') and args.si_c>0) else ""
        if verbose and checkattr(args, 'ewc') and args.ewc_lambda>0:
            print(" --> EWC:           " + ewc_stamp)
        if verbose and checkattr(args, 'si') and args.si_c>0:
            print(" --> SI:            " + si_stamp)
    ewc_stamp = "--{}{}{}".format(ewc_stamp, both, si_stamp) if (
            (checkattr(args, 'ewc') and args.ewc_lambda>0) or (checkattr(args, 'si') and args.si_c>0)
    ) else ""

    # -for XdG
    xdg_stamp = ""
    if (checkattr(args, "xdg") and args.xdg_prop > 0):
        xdg_stamp = "--XdG{}".format(args.xdg_prop)
        if verbose:
            print(" --> XdG:           " + "gating = {}".format(args.xdg_prop))

    # -for replay
    if replay:
        replay_stamp = "{rep}{bat}{agem}{distil}".format(
            rep=args.replay,
            bat="" if (
                    (not hasattr(args, 'batch_replay')) or (args.batch_replay is None) or args.batch_replay==args.batch
            ) else "-br{}".format(args.batch_replay),
            agem="-aGEM" if args.agem else "",
            distil="-Di{}".format(args.temp) if args.distill else "",
        )
        if verbose:
            print(" --> replay:        " + replay_stamp)
    replay_stamp = "--{}".format(replay_stamp) if replay else ""

    # -for exemplars
    exemplar_stamp = ""
    if checkattr(args, 'use_exemplars') or (hasattr(args, 'replay') and args.replay=="exemplars"):
        exemplar_opts = "b{}{}".format(args.budget, "H" if args.herding else "")
        use = "useEx-" if args.use_exemplars else ""
        exemplar_stamp = "--{}{}".format(use, exemplar_opts)
        if verbose:
            print(" --> exemplars:     " + "{}{}".format(use, exemplar_opts))

    # --> combine
    param_stamp = "{}--{}--{}{}{}{}{}{}".format(
        task_stamp, model_stamp, hyper_stamp, ewc_stamp, xdg_stamp, replay_stamp, exemplar_stamp,
        "-s{}".format(args.seed) if not args.seed==0 else "",
    )
    reinit_param_stamp = "{}--{}--{}{}{}".format(
        task_stamp, model_stamp, hyper_stamp, "-R" if not checkattr(args, 'reinit') else "",
        "-s{}".format(args.seed) if not args.seed==0 else "",
    )

    ## Print param-stamp on screen and return
    if verbose:
        print(param_stamp)
    return param_stamp, reinit_param_stamp