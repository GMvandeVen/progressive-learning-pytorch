import numpy as np
import torch
from torch.nn import functional as F
from models.utils import loss_functions as lf, modules
from models.conv.nets import ConvLayers
from models.fc.layers import fc_layer
from models.fc.nets import MLP
from models.cl.continual_learner import ContinualLearner
from models.cl.exemplars import ExemplarHandler


class Classifier(ContinualLearner, ExemplarHandler):
    '''Model for encoding (i.e., feature extraction) and classifying images, enriched as "ContinualLearner"--object.'''

    def __init__(self, image_size, image_channels, classes,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=1000, h_dim=400, fc_drop=0, fc_bn=True, fc_nl="relu", fc_gated=False,
                 bias=True, excitability=False, excit_buffer=False,
                 # -training-related parameters
                 AGEM=False):

        # model configurations
        super().__init__()
        self.classes = classes
        self.label = "Classifier"
        self.depth = depth
        self.fc_layers = fc_layers
        self.fc_drop = fc_drop

        # settings for training
        self.AGEM = AGEM  #-> use gradient of replayed data as inequality constraint for (instead of adding it to)
                          #   the gradient of the current data (as in A-GEM, see Chaudry et al., 2019; ICLR)

        # optimizer (needs to be set before training starts))
        self.optimizer = None
        self.optim_list = []

        # check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")

        ######------SPECIFY MODEL------######
        #--> convolutional layers
        self.convE = ConvLayers(
            conv_type=conv_type, block_type="basic", num_blocks=num_blocks, image_channels=image_channels,
            depth=depth, start_channels=start_channels, reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl,
            global_pooling=global_pooling, gated=conv_gated, output="none" if no_fnl else "normal",
        )
        self.flatten = modules.Flatten()
        #------------------------------calculate input/output-sizes--------------------------------#
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels
        if fc_layers<2:
            self.fc_layer_sizes = [self.conv_out_units]  #--> this results in self.fcE = modules.Identity()
        elif fc_layers==2:
            self.fc_layer_sizes = [self.conv_out_units, h_dim]
        else:
            self.fc_layer_sizes = [self.conv_out_units]+[int(x) for x in np.linspace(fc_units, h_dim, num=fc_layers-1)]
        self.units_before_classifier = h_dim if fc_layers>1 else self.conv_out_units
        #------------------------------------------------------------------------------------------#
        #--> fully connected layers
        self.fcE = MLP(size_per_layer=self.fc_layer_sizes, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                       excitability=excitability, excit_buffer=excit_buffer, gated=fc_gated)#, output="none") ## NOTE: temporary change!!!
        #--> classifier
        self.classifier = fc_layer(self.units_before_classifier, classes, excit_buffer=True, nl='none', drop=fc_drop)


    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        if self.depth>0 and self.fc_layers>1:
            return "{}_{}_c{}".format(self.convE.name, self.fcE.name, self.classes)
        elif self.depth>0:
            return "{}_{}c{}".format(self.convE.name, "drop{}-".format(self.fc_drop) if self.fc_drop>0 else "",
                                     self.classes)
        elif self.fc_layers>1:
            return "{}_c{}".format(self.fcE.name, self.classes)
        else:
            return "i{}_{}c{}".format(self.fc_layer_sizes[0], "drop{}-".format(self.fc_drop) if self.fc_drop>0 else "",
                                      self.classes)


    def forward(self, x):
        hidden_rep = self.convE(x)
        final_features = self.fcE(self.flatten(hidden_rep))
        return self.classifier(final_features)

    def feature_extractor(self, images):
        return self.fcE(self.flatten(self.convE(images)))

    def classify(self, x):
        '''For input [x] (image or extracted "intermediate" image features), return all predicted "scores"/"logits".'''
        image_features = self.flatten(self.convE(x))
        hE = self.fcE(image_features)
        return self.classifier(hE)


    def train_a_batch(self, x, y=None, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None,
                      task=1, freeze_convE=False, **kwargs):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_]).

        [x]                 <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]                 <tensor> batch of corresponding labels
        [x_]                None or (<list> of) <tensor> batch of replayed inputs
        [y_]                None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [scores_]           None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]               <number> in [0,1], relative importance of new task
        [active_classes]    None or (<list> of) <list> with "active" classes
        [task]              <int>, for setting task-specific mask'''

        # Set model to training-mode
        self.train()
        if freeze_convE:
            # - if conv-layers are frozen, they shoud be set to eval() to prevent batch-norm layers from changing
            self.convE.eval()

        # Reset optimizer
        self.optimizer.zero_grad()

        # Should gradient be computed separately for each task? (needed when a task-mask is combined with replay)
        gradient_per_task = True if ((self.mask_dict is not None) and (x_ is not None)) else False


        ##--(1)-- REPLAYED DATA --##

        if x_ is not None:
            # In the Task-IL scenario, [y_] or [scores_] is a list and [x_] needs to be evaluated on each of them
            # (in case of 'exact' or 'exemplar' replay, [x_] is also a list!
            TaskIL = (type(y_)==list) if (y_ is not None) else (type(scores_)==list)
            if not TaskIL:
                y_ = [y_]
                scores_ = [scores_]
                active_classes = [active_classes] if (active_classes is not None) else None
            n_replays = len(y_) if (y_ is not None) else len(scores_)

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays
            predL_r = [None]*n_replays
            distilL_r = [None]*n_replays

            # Run model (if [x_] is not a list with separate replay per task and there is no task-specific mask)
            if (not type(x_)==list) and (self.mask_dict is None):
                y_hat_all = self(x_)

            # Loop to evalute predictions on replay according to each previous task
            for replay_id in range(n_replays):

                # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                if (type(x_)==list) or (self.mask_dict is not None):
                    x_temp_ = x_[replay_id] if type(x_)==list else x_
                    if self.mask_dict is not None:
                        self.apply_XdGmask(task=replay_id+1)
                    y_hat_all = self(x_temp_)

                # -if needed, remove predictions for classes not in replayed task
                y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]

                # Calculate losses
                if (y_ is not None) and (y_[replay_id] is not None):
                    predL_r[replay_id] = F.cross_entropy(y_hat, y_[replay_id], reduction='mean')
                if (scores_ is not None) and (scores_[replay_id] is not None):
                    # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes are added to [scores]!
                    n_classes_to_consider = y_hat.size(1)    #--> zeros will be added to [scores] to make it this size!
                    kd_fn = lf.loss_fn_kd
                    distilL_r[replay_id] = kd_fn(scores=y_hat[:, :n_classes_to_consider],
                                                 target_scores=scores_[replay_id], T=self.KD_temp)
                # Weigh losses
                if self.replay_targets=="hard":
                    loss_replay[replay_id] = predL_r[replay_id]
                elif self.replay_targets=="soft":
                    loss_replay[replay_id] = distilL_r[replay_id]

                # If needed, perform backward pass before next task-mask (gradients of all tasks will be accumulated)
                if gradient_per_task:
                    weight = 1 if self.AGEM else (1-rnt)
                    weighted_replay_loss_this_task = weight * loss_replay[replay_id] / n_replays
                    weighted_replay_loss_this_task.backward()

        # Calculate total replay loss
        loss_replay = None if (x_ is None) else sum(loss_replay)/n_replays

        # If using A-GEM, calculate and store averaged gradient of replayed data
        if self.AGEM and x_ is not None:
            # Perform backward pass to calculate gradient of replayed batch (if not yet done)
            if not gradient_per_task:
                loss_replay.backward()
            # Reorganize the gradient of the replayed batch as a single vector
            grad_rep = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_rep.append(p.grad.view(-1))
            grad_rep = torch.cat(grad_rep)
            # Reset gradients (with GEM, gradients of replayed batch should only be used as inequality constraint)
            self.optimizer.zero_grad()


        ##--(2)-- CURRENT DATA --##

        if x is not None:
            # If requested, apply correct task-specific mask
            if self.mask_dict is not None:
                self.apply_XdGmask(task=task)

            # Run model
            y_hat = self(x)
            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                y_hat = y_hat[:, class_entries]

            # Calculate prediction loss
            predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='mean')

            # Weigh losses
            loss_cur = predL

            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)

            # If backward passes are performed per task (i.e., when XdG combined with replay), perform backward pass
            if gradient_per_task:
                weighted_current_loss = rnt*loss_cur
                weighted_current_loss.backward()
        else:
            precision = predL = None
            # -> it's possible there is only "replay" [i.e., for offline with incremental task learning]


        # Combine loss from current and replayed batch
        if x_ is None or self.AGEM:
            loss_total = loss_cur
        else:
            loss_total = loss_replay if (x is None) else rnt*loss_cur+(1-rnt)*loss_replay


        ##--(3)-- ALLOCATION LOSSES --##

        # Add SI-loss (Zenke et al., 2017)
        surrogate_loss = self.surrogate_loss()
        if self.si_c>0:
            loss_total += self.si_c * surrogate_loss

        # Add EWC-loss
        ewc_loss = self.ewc_loss()
        if self.ewc_lambda>0:
            loss_total += self.ewc_lambda * ewc_loss


        # Backpropagate errors (if not yet done)
        if not gradient_per_task:
            loss_total.backward()

        # If using A-GEM:
        if self.AGEM and x_ is not None:
            # -reorganize gradient (of current batch) as single vector
            grad_cur = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            # -check inequality constrain
            angle = (grad_cur * grad_rep).sum()
            if angle < 0:
                # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                length_rep = (grad_rep * grad_rep).sum()
                grad_proj = grad_cur - (angle / length_rep) * grad_rep
                # -...and replace all the gradients within the model with this projected gradient
                index = 0
                for p in self.parameters():
                    if p.requires_grad:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index:index + n_param].view_as(p))
                        index += n_param

        # Take optimization-step
        self.optimizer.step()


        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
            'pred': predL.item() if predL is not None else 0,
            'pred_r': sum(predL_r).item()/n_replays if (x_ is not None and predL_r[0] is not None) else 0,
            'distil_r': sum(distilL_r).item()/n_replays if (x_ is not None and distilL_r[0] is not None) else 0,
            'ewc': ewc_loss.item(), 'si_loss': surrogate_loss.item(),
            'precision': precision if precision is not None else 0.,
        }

