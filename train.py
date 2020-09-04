import numpy as np
import torch
from torch import optim
import tqdm
import copy
import utils
from data.manipulate import SubDataset, ExemplarDataset
from models.cl.continual_learner import ContinualLearner


def train(model, train_loader, iters, loss_cbs=list(), eval_cbs=list(), save_every=None, m_dir="./store/models",
          args=None):
    '''Train a model with a "train_a_batch" method for [iters] iterations on data from [train_loader].

    [model]             model to optimize
    [train_loader]      <dataloader> for training [model] on
    [iters]             <int> (max) number of iterations (i.e., batches) to train for
    [loss_cbs]          <list> of callback-<functions> to keep track of training progress
    [eval_cbs]          <list> of callback-<functions> to evaluate model on separate data-set'''

    device = model._device()

    # Should convolutional layers be frozen?
    freeze_convE = (utils.checkattr(args, "freeze_convE") and hasattr(args, "depth") and args.depth>0)

    # Create progress-bar (with manual control)
    bar = tqdm.tqdm(total=iters)

    iteration = epoch = 0
    while iteration < iters:
        epoch += 1

        # Loop over all batches of an epoch
        for batch_idx, (data, y) in enumerate(train_loader):
            iteration += 1

            # Perform training-step on this batch
            data, y = data.to(device), y.to(device)
            loss_dict = model.train_a_batch(data, y=y, freeze_convE=freeze_convE)

            # Fire training-callbacks (for visualization of training-progress)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(bar, iteration, loss_dict, epoch=epoch)

            # Fire evaluation-callbacks (to be executed every [eval_log] iterations, as specified within the functions)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, iteration, epoch=epoch)

            # Break if max-number of iterations is reached
            if iteration == iters:
                bar.close()
                break

            # Save checkpoint?
            if (save_every is not None) and (iteration % save_every) == 0:
                utils.save_checkpoint(model, model_dir=m_dir)



def train_cl(model, train_datasets, replay_mode="none", rnt=None, classes_per_task=None,
             iters=2000, batch_size=32, batch_size_replay=None, loss_cbs=list(), eval_cbs=list(), reinit=False,
             args=None, only_last=False, use_exemplars=False, metric_cbs=list()):
    '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]             <nn.Module> main model to optimize across all tasks
    [train_datasets]    <list> with for each task the training <DataSet>
    [replay_mode]       <str>, choice from "current", "offline" and "none"
    [classes_per_task]  <int>, # classes per task; only 1st task has [classes_per_task]*[first_task_class_boost] classes
    [rnt]               <float>, indicating relative importance of new task (if None, relative to # old tasks)
    [iters]             <int>, # optimization-steps (=batches) per task; 1st task has [first_task_iter_boost] steps more
    [batch_size_replay] <int>, number of samples to replay per batch
    [only_last]         <bool>, only train on final task / episode
    [*_cbs]             <list> of call-back functions to evaluate training-progress'''

    # Should convolutional layers be frozen?
    freeze_convE = (utils.checkattr(args, "freeze_convE") and hasattr(args, "depth") and args.depth>0)

    # Use cuda?
    device = model._device()
    cuda = model._is_on_cuda()

    # Set default-values if not specified
    batch_size_replay = batch_size if batch_size_replay is None else batch_size_replay

    # Initiate indicators for replay (no replay for 1st task)
    Exact = Current = Offline_TaskIL = False
    previous_model = None

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and model.si_c>0:
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())

    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):

        # In offline replay-setting, all tasks so far should be visited separately (i.e., separate data-loader per task)
        if replay_mode=="offline":
            Offline_TaskIL = True
            data_loader = [None]*task

        train_dataset = train_dataset

        # Initialize # iters left on data-loader(s)
        iters_left = 1 if (not Offline_TaskIL) else [1]*task
        if Exact:
            iters_left_previous = [1]*(task-1)
            data_loader_previous = [None]*(task-1)

        # Prepare <dicts> to store running importance estimates and parameter-values before update
        if isinstance(model, ContinualLearner) and model.si_c>0:
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

        # Find [active_classes] (=classes in current task)
        active_classes = [list(range(classes_per_task*i, classes_per_task*(i+1))) for i in range(task)]

        # Reinitialize the model's parameters and the optimizer (if requested)
        if reinit:
            from define_models import init_params
            init_params(model, args)
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # Define a tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))

        # Loop over all iterations
        iters_to_use = iters
        # -if only the final task should be trained on:
        if only_last and not task==len(train_datasets):
            iters_to_use = 0
        for batch_index in range(1, iters_to_use+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            if not Offline_TaskIL:
                iters_left -= 1
                if iters_left==0:
                    data_loader = iter(utils.get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=True))
                    # NOTE:  [train_dataset]  is training-set of current task
                    #      [train_dataset] is training-set of current task with stored exemplars added (if requested)
                    iters_left = len(data_loader)
            else:
                # -with "offline replay", there is a separate data-loader for each task
                batch_size_to_use = batch_size
                for task_id in range(task):
                    iters_left[task_id] -= 1
                    if iters_left[task_id]==0:
                        data_loader[task_id] = iter(utils.get_data_loader(
                            train_datasets[task_id], batch_size_to_use, cuda=cuda, drop_last=True
                        ))
                        iters_left[task_id] = len(data_loader[task_id])

            # Update # iters left on data-loader(s) of the previous task(s) and, if needed, create new one(s)
            if Exact:
                up_to_task = task-1
                batch_size_replay_pt = int(np.floor(batch_size_replay/up_to_task)) if (up_to_task>1) else batch_size_replay
                # -need separate replay for each task
                for task_id in range(up_to_task):
                    batch_size_to_use = min(batch_size_replay_pt, len(previous_datasets[task_id]))
                    iters_left_previous[task_id] -= 1
                    if iters_left_previous[task_id]==0:
                        data_loader_previous[task_id] = iter(utils.get_data_loader(
                            previous_datasets[task_id], batch_size_to_use, cuda=cuda, drop_last=True
                        ))
                        iters_left_previous[task_id] = len(data_loader_previous[task_id])



            #-----------------Collect data------------------#

            #####-----CURRENT BATCH-----#####
            if not Offline_TaskIL:
                x, y = next(data_loader)                        #--> sample training data of current task
                y = y-classes_per_task*(task-1)                 #--> ITL: adjust y-targets to 'active range'
                x, y = x.to(device), y.to(device)               #--> transfer them to correct device
                #y = y.expand(1) if len(y.size())==1 else y     #--> hack for if batch-size is 1
            else:
                x = y = task_used = None  #--> all tasks are "treated as replay"
                # -sample training data for all tasks so far, move to correct device and store in lists
                x_, y_ = list(), list()
                for task_id in range(task):
                    x_temp, y_temp = next(data_loader[task_id])
                    x_.append(x_temp.to(device))
                    y_temp = y_temp - (classes_per_task * task_id) #--> adjust y-targets to 'active range'
                    if batch_size_to_use == 1:
                        y_temp = torch.tensor([y_temp])            #--> correct dimensions if batch-size is 1
                    y_.append(y_temp.to(device))


            #####-----REPLAYED BATCH-----#####
            if not Offline_TaskIL and not Exact and not Current:
                x_ = y_ = scores_ = task_used = None   #-> if no replay

            #--------------------------------------------INPUTS----------------------------------------------------#

            ##-->> Exact Replay <<--##
            if Exact:
                # Sample replayed training data, move to correct device and store in lists
                x_ = list()
                y_ = list()
                up_to_task = task-1
                for task_id in range(up_to_task):
                    x_temp, y_temp = next(data_loader_previous[task_id])
                    x_.append(x_temp.to(device))
                    # -only keep [y_] if required (as otherwise unnecessary computations will be done)
                    if model.replay_targets=="hard":
                        y_temp = y_temp - (classes_per_task*task_id) #-> adjust y-targets to 'active range'
                        y_.append(y_temp.to(device))
                    else:
                        y_.append(None)
                # If required, get target scores (i.e, [scores_])        -- using previous model, with no_grad()
                if (model.replay_targets=="soft") and (previous_model is not None):
                    scores_ = list()
                    for task_id in range(up_to_task):
                        with torch.no_grad():
                            scores_temp = previous_model(x_[task_id])
                        scores_temp = scores_temp[:, (classes_per_task*task_id):(classes_per_task*(task_id+1))]
                        scores_.append(scores_temp)
                else:
                    scores_ = None

            ##-->> Current Replay <<--##
            if Current:
                x_ = x[:batch_size_replay]  #--> use current task inputs
                task_used = None


            #--------------------------------------------OUTPUTS----------------------------------------------------#

            if Current:
                # Get target scores & possibly labels (i.e., [scores_] / [y_]) -- use previous model, with no_grad()
                # -[x_] needs to be evaluated according to each previous task, so make list with entry per task
                scores_ = list()
                y_ = list()
                # -if no task-mask and no conditional generator, all scores can be calculated in one go
                if previous_model.mask_dict is None and not type(x_)==list:
                    with torch.no_grad():
                        all_scores_ = previous_model.classify(x_)
                for task_id in range(task-1):
                    # -if there is a task-mask (i.e., XdG is used), obtain predicted scores for each task separately
                    if previous_model.mask_dict is not None:
                        previous_model.apply_XdGmask(task=task_id+1)
                    if previous_model.mask_dict is not None or type(x_)==list:
                        with torch.no_grad():
                            all_scores_ = previous_model.classify(x_[task_id] if type(x_)==list else x_)
                    temp_scores_ = all_scores_[:, (classes_per_task*task_id):(classes_per_task*(task_id+1))]
                    scores_.append(temp_scores_)
                    # - also get hard target
                    _, temp_y_ = torch.max(temp_scores_, dim=1)
                    y_.append(temp_y_)
            # -only keep predicted y_/scores_ if required (as otherwise unnecessary computations will be done)
            y_ = y_ if (model.replay_targets=="hard") else None
            scores_ = scores_ if (model.replay_targets=="soft") else None


            #-----------------Train model------------------#

            # Train the main model with this batch
            loss_dict = model.train_a_batch(x, y=y, x_=x_, y_=y_, scores_=scores_,
                                            tasks_=task_used, active_classes=active_classes, task=task, rnt=(
                                                1. if task==1 else 1./task
                                            ) if rnt is None else rnt, freeze_convE=freeze_convE)

            # Update running parameter importance estimates in W
            if isinstance(model, ContinualLearner) and model.si_c>0:
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            W[n].add_(-p.grad*(p.detach()-p_old[n]))
                        p_old[n] = p.detach().clone()

            # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, batch_index, loss_dict, task=task)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, batch_index, task=task)


        # Close progres-bar
        progress.close()


        ##----------> UPON FINISHING EACH TASK...

        # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty
        if isinstance(model, ContinualLearner) and model.ewc_lambda>0:
            # -find allowed classes
            allowed_classes = list(range(classes_per_task*(task-1), classes_per_task*task))
            # -if needed, apply correct task-specific mask
            if model.mask_dict is not None:
                model.apply_XdGmask(task=task)
            # -estimate FI-matrix
            model.estimate_fisher(train_dataset, allowed_classes=allowed_classes)

        # SI: calculate and update the normalized path integral
        if isinstance(model, ContinualLearner) and model.si_c>0:
            model.update_omega(W, model.epsilon)

        # EXEMPLARS: update exemplar sets
        if use_exemplars or replay_mode=="exemplars":
            exemplars_per_class = int(np.floor(model.memory_budget / (classes_per_task*task)))
            # reduce examplar-sets
            model.reduce_exemplar_sets(exemplars_per_class)
            # for each new class trained on, construct examplar-set
            new_classes = list(range(classes_per_task*(task-1), classes_per_task*task))
            for class_id in new_classes:
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                # based on this dataset, construct new exemplar-set for this class
                model.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class)
            model.compute_means = True

        # Calculate statistics required for metrics
        for metric_cb in metric_cbs:
            if metric_cb is not None:
                metric_cb(model, iters, task=task)

        # REPLAY: update source for replay
        previous_model = copy.deepcopy(model).eval()
        if replay_mode=='current':
            Current = True
        elif replay_mode in ('exemplars', 'exact'):
            Exact = True
            if replay_mode=="exact":
                previous_datasets = train_datasets[:task]
            else:
                previous_datasets = []
                for task_id in range(task):
                    previous_datasets.append(
                        ExemplarDataset(
                            model.exemplar_sets[
                            (classes_per_task * task_id):(classes_per_task * (task_id + 1))],
                            target_transform=lambda y, x=classes_per_task * task_id: y + x)
                    )
