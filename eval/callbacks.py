import visual.visdom
from . import evaluate


#########################################################
## Callback-functions for evaluating model-performance ##
#########################################################


def _eval_cb(log, test_datasets, visdom=None, iters_per_task=None, test_size=None,
             classes_per_task=None, with_exemplars=False):
    '''Initiates function for evaluating performance of classifier (in terms of precision).

    [test_datasets]     <list> of <Datasets>; also if only 1 task, it should be presented as a list!
    [classes_per_task]  <int> number of "active" classes per task
    '''

    def eval_cb(classifier, batch, task=1, **kwargs):
        '''Callback-function, to evaluate performance of classifier.'''

        iteration = batch if task==1 else (task-1)*iters_per_task + batch

        # evaluate the solver on multiple tasks (and log to visdom)
        if iteration % log == 0:
            evaluate.precision(classifier, test_datasets, task, iteration,
                               classes_per_task=classes_per_task,
                               test_size=test_size, visdom=visdom, with_exemplars=with_exemplars)

    ## Return the callback-function (except if neither visdom or [precision_dict] is selected!)
    return eval_cb if (visdom is not None) else None



##------------------------------------------------------------------------------------------------------------------##

################################################
## Callback-functions for calculating metrics ##
################################################

def _metric_cb(log, test_datasets, metrics_dict=None, iters_per_task=None, test_size=None, classes_per_task=None,
               with_exemplars=False):
    '''Initiates function for calculating statistics required for calculating metrics.

    [test_datasets]     <list> of <Datasets>; also if only 1 task, it should be presented as a list!
    [classes_per_task]  <int> number of "active" classes per task
    '''

    def metric_cb(classifier, batch, task=1):
        '''Callback-function, to calculate statistics for metrics.'''

        iteration = batch if task==1 else (task-1)*iters_per_task + batch

        # evaluate the solver on multiple tasks (and log to visdom)
        if iteration % log == 0:
            evaluate.metric_statistics(classifier, test_datasets, task, iteration,
                                       classes_per_task=classes_per_task, metrics_dict=metrics_dict,
                                       test_size=test_size, with_exemplars=with_exemplars)

    ## Return the callback-function (except if no [metrics_dict] is selected!)
    return metric_cb if (metrics_dict is not None) else None



##------------------------------------------------------------------------------------------------------------------##

###############################################################
## Callback-functions for keeping track of training-progress ##
###############################################################

def _solver_loss_cb(log, visdom, model=None, tasks=None, iters_per_task=None, epochs=None, rnt=None, replay=False,
                    progress_bar=True):
    '''Initiates function for keeping track of, and reporting on, the progress of the solver's training.'''

    def cb(bar, iter, loss_dict, task=1, epoch=None):
        '''Callback-function, to call on every iteration to keep track of training progress.'''

        iteration = iter if task==1 else (task-1)*iters_per_task + iter

        ##--------------------------------PROGRESS BAR---------------------------------##
        if progress_bar and bar is not None:
            task_stm = "" if (tasks is None) else " Task: {}/{} |".format(task, tasks)
            epoch_stm = "" if ((epochs is None) or (epoch is None)) else " Epoch: {}/{} |".format(epoch, epochs)
            bar.set_description(
                ' <MAIN MODEL> |{t_stm}{e_stm} training loss: {loss:.3} | training precision: {prec:.3} |'
                    .format(t_stm=task_stm, e_stm=epoch_stm, loss=loss_dict['loss_total'], prec=loss_dict['precision'])
            )
            bar.update(1)
        ##-----------------------------------------------------------------------------##

        # log the loss of the solver (to visdom)
        if (iteration % log == 0) and (visdom is not None):
            if tasks is None or tasks==1:
                # -overview of loss -- single task
                plot_data = [loss_dict['pred']]
                names = ['prediction']
            else:
                # -overview of losses -- multiple tasks
                current_rnt = (1./task if (rnt is None) or (task==1) else rnt) if replay else 1.
                plot_data = [current_rnt*loss_dict['pred']]
                names = ['pred']
                if replay:
                    if model.replay_targets=="hard":
                        plot_data += [(1-current_rnt)*loss_dict['pred_r']]
                        names += ['pred - r']
                    elif model.replay_targets=="soft":
                        plot_data += [(1-current_rnt)*loss_dict['distil_r']]
                        names += ['KD - r']
                if model.ewc_lambda>0:
                    plot_data += [model.ewc_lambda * loss_dict['ewc']]
                    names += ['EWC (lambda={})'.format(model.ewc_lambda)]
                if model.si_c>0:
                    plot_data += [model.si_c * loss_dict['si_loss']]
                    names += ['SI (c={})'.format(model.si_c)]
            visual.visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iteration,
                title="CLASSIFIER: loss ({})".format(visdom["graph"]), env=visdom["env"], ylabel="training loss"
            )

    # Return the callback-function.
    return cb