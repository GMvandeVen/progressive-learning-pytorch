import torch
import visual.visdom
import visual.plt
import utils


####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####


def validate(model, dataset, batch_size=128, test_size=1024, verbose=True, allowed_classes=None,
             no_task_mask=False, task=None, with_exemplars=False):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Get device-type / using cuda?
    device = model._device()
    cuda = model._is_on_cuda()

    # Set model to eval()-mode
    model.eval()

    # Apply task-specifc "gating-mask" for each hidden fully connected layer (or remove it!)
    if hasattr(model, "mask_dict") and model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = total_correct = 0
    for data, labels in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -evaluate model (if requested, only on [allowed_classes])
        data, labels = data.to(device), labels.to(device)
        labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels
        with torch.no_grad():
            if with_exemplars:
                predicted = model.classify_with_exemplars(data, allowed_classes=allowed_classes)
            else:
                scores = model.classify(data)
                scores = scores if (allowed_classes is None) else scores[:, allowed_classes]
                _, predicted = torch.max(scores, 1)
        # -update statistics
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)
    precision = total_correct / total_tested

    # Print result on screen (if requested) and return it
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision


def precision(model, datasets, current_task, iteration, classes_per_task=None,
              test_size=None, visdom=None, verbose=False, summary_graph=True, with_exemplars=False, no_task_mask=False):
    '''Evaluate precision of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [classes_per_task]  <int> number of active classes er task
    [visdom]            None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

    n_tasks = len(datasets)

    # Evaluate accuracy of model predictions for all tasks so far (reporting "0" for future tasks)
    precs = []
    for i in range(n_tasks):
        if (i+1 <= current_task):
            allowed_classes = list(range(classes_per_task * i, classes_per_task * (i + 1)))
            precs.append(validate(model, datasets[i], test_size=test_size, verbose=verbose,
                                  allowed_classes=allowed_classes, with_exemplars=with_exemplars,
                                  no_task_mask=no_task_mask, task=i + 1))
        else:
            precs.append(0)
    average_precs = sum([precs[task_id] for task_id in range(current_task)]) / current_task

    # Print results on screen
    if verbose:
        print(' => ave precision: {:.3f}'.format(average_precs))

    # Send results to visdom server
    names = ['task {}'.format(i + 1) for i in range(n_tasks)]
    if visdom is not None:
        visual.visdom.visualize_scalars(
            precs, names=names, title="precision ({})".format(visdom["graph"]),
            iteration=iteration, env=visdom["env"], ylabel="test precision"
        )
        if n_tasks>1 and summary_graph:
            visual.visdom.visualize_scalars(
                [average_precs], names=["ave"], title="ave precision ({})".format(visdom["graph"]),
                iteration=iteration, env=visdom["env"], ylabel="test precision"
            )



####--------------------------------------------------------------------------------------------------------------####

####---------------------------####
####----METRIC CALCULATIONS----####
####---------------------------####


def initiate_metrics_dict(n_tasks):
    '''Initiate <dict> with all measures to keep track of.'''
    metrics_dict = {}
    metrics_dict["average"] = []     # ave acc over all tasks so far: Task-IL -> only classes in task
                                     #                                Class-IL-> all classes so far (up to trained task)
    metrics_dict["x_iteration"] = [] # total number of iterations so far
    metrics_dict["x_task"] = []      # number of tasks so far (indicating the task on which training just finished)
    # Accuracy matrix
    metrics_dict["acc per task"] = {}
    for i in range(n_tasks):
        metrics_dict["acc per task"]["task {}".format(i+1)] = []
    return metrics_dict


def intial_accuracy(model, datasets, metrics_dict, classes_per_task=None, test_size=None,
                    verbose=False, no_task_mask=False):
    '''Evaluate precision of a classifier (=[model]) on all tasks using [datasets] before any learning.'''

    n_tasks = len(datasets)
    precs = []

    for i in range(n_tasks):
        precision = validate(
            model, datasets[i], test_size=test_size, verbose=verbose,
            allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))),
            no_task_mask=no_task_mask, task=i+1
        )
        precs.append(precision)

    metrics_dict["initial acc per task"] = precs
    return metrics_dict


def metric_statistics(model, datasets, current_task, iteration, classes_per_task=None,
                      metrics_dict=None, test_size=None, verbose=False, with_exemplars=False, no_task_mask=False):
    '''Evaluate precision of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [metrics_dict]      None or <dict> of all measures to keep track of, to which results will be appended to
    [classes_per_task]  <int> number of active classes er task
    '''

    n_tasks = len(datasets)

    # Calculate accurcies per task
    precs = []
    for i in range(n_tasks):
        allowed_classes = list(range(classes_per_task * i, classes_per_task * (i + 1)))
        precision = validate(model, datasets[i], test_size=test_size, verbose=verbose,
                             allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1,
                             with_exemplars=with_exemplars) if (not with_exemplars) or (i<current_task) else 0.
        precs.append(precision)

    # Calcualte average accuracy over all tasks thus far
    average_precs = sum([precs[task_id] for task_id in range(current_task)]) / current_task

    # Append results to [metrics_dict]-dictionary
    for task_id in range(n_tasks):
        metrics_dict["acc per task"]["task {}".format(task_id+1)].append(precs[task_id])
    metrics_dict["average"].append(average_precs)
    metrics_dict["x_iteration"].append(iteration)
    metrics_dict["x_task"].append(current_task)

    # Print results on screen
    if verbose:
        print(' => ave precision: {:.3f}'.format(average_precs))

    return metrics_dict
