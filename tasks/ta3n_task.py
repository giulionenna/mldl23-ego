from abc import ABC
import torch
from utils import utils
from functools import reduce
import wandb
import tasks
from utils.logger import logger
from torch import nn
from typing import Dict, Tuple


class TA3N_task(tasks.Task, ABC):
    """Ta3N."""
    
    def __init__(self, name: str, task_models: Dict[str, torch.nn.Module], batch_size: int, 
                 total_batch: int, models_dir: str, num_classes: int,
                 num_clips: int, model_args: Dict[str, float], args,device, **kwargs) -> None:
        """Create an instance of the action recognition model.

        Parameters
        ----------
        name : str
            name of the task e.g. action_classifier, domain_classifier...
        task_models : Dict[str, torch.nn.Module]
            torch models, one for each different modality adopted by the task
        batch_size : int
            actual batch size in the forward
        total_batch : int
            batch size simulated via gradient accumulation
        models_dir : str
            directory where the models are stored when saved
        num_classes : int
            number of labels in the classification task
        num_clips : int
            number of clips
        model_args : Dict[str, float]
            model-specific arguments
        """
        super().__init__(name, task_models, batch_size, total_batch, models_dir, args, **kwargs)
        self.model_args = model_args

        # self.accuracy and self.loss track the evolution of the accuracy and the training loss
        self.accuracy = utils.Accuracy(topk=(1, 5), classes=num_classes)
        self.accuracy_class = utils.Accuracy(topk=(1, 5), classes=num_classes)
        self.accuracy_td = utils.Accuracy(classes=2)
        self.accuracy_sd = utils.Accuracy(classes=2)
        
        self.loss = utils.AverageMeter()
    
        self.loss_class = utils.AverageMeter() ########## controlla ##########
        self.loss_td = utils.AverageMeter()
        self.loss_sd = utils.AverageMeter()
        self.loss_rd = utils.AverageMeter()
        self.loss_ae = utils.AverageMeter()


        
        self.num_clips = num_clips
        self.batch_size = batch_size
        self.device = device

        self.gamma = model_args['RGB'].gamma
        self.l_s = model_args['RGB'].l_s
        self.l_r = model_args['RGB'].l_r
        self.l_t = model_args['RGB'].l_t

        # Use the cross entropy loss as the default criterion for the classification task
        self.criterion_class = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='none')
        self.criterion_td = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='none')
        self.criterion_sd = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='none')
        self.criterion_rd = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='none')
        # Initializeq the model parameters and the optimizer
        optim_params = {}
        self.optimizer = dict()
        for m in self.modalities:
            optim_params[m] = filter(lambda parameter: parameter.requires_grad, self.task_models[m].parameters())
            self.optimizer[m] = torch.optim.SGD(optim_params[m], model_args[m].lr,
                                                weight_decay=model_args[m].weight_decay,
                                                momentum=model_args[m].sgd_momentum)

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward step of the task

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            a dictionary that stores the input data for each modality 

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            output logits and features
        """
        # logits_class = {}
        # logits_td = {}
        # logits_sd = {}
        logits = {"class":{},
                  "td":{},
                  "sd":{},
                  "rd":{}}
        ""

        features = {}
        for i_m, m in enumerate(self.modalities):
            logits["sd"][m],logits["td"][m],logits["class"][m],logits["rd"][m]= self.task_models[m](x=data[m], **kwargs)

        return logits

    def compute_loss(self, logits, label_class: torch.Tensor,label_d: torch.Tensor, loss_weight: float=1.0,domain = "source"):
        """Fuse the logits from different modalities and compute the classification loss.

        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        loss_weight : float, optional
            weight of the classification loss, by default 1.0
        """
        domain_entropy = 0; # ablation['td'] = False
        eps = 1e-7
        if(domain == "source"): #source
            loss = 0
            #Compute Class loss
            fused_logits_class = reduce(lambda x, y: x + y, logits["class"].values())
            self.loss_class.update(self.criterion_class(fused_logits_class, label_class))
            loss +=  self.loss_class.val

            if(self.model_args['RGB']["ablation"]["gsd"]):
                fused_logits_sd = reduce(lambda x, y: x + y, logits["sd"].values())
                self.loss_sd.update(self.criterion_sd(fused_logits_sd, label_d))
                loss +=  0.5*self.loss_sd.val
            if(self.model_args['RGB']["ablation"]["gtd"]):    
                fused_logits_td = reduce(lambda x, y: x + y, logits["td"].values())
                self.loss_td.update(self.criterion_td(fused_logits_td, label_d))
                loss +=  0.5*self.loss_td.val
                
                # Loss AE
                softmax = nn.Softmax(dim=1)
                logsoftmax = nn.LogSoftmax(dim=1)
                #domain_entropy = torch.sum(-(fused_logits_td) * torch.log(fused_logits_td+eps), 1)
                #class_entropy = torch.sum(-(fused_logits_class) * torch.log(fused_logits_class+eps), 1) 

                domain_entropy = torch.sum(-softmax(fused_logits_td) * logsoftmax(fused_logits_td+eps), 1)
                class_entropy = torch.sum(-softmax(fused_logits_class) * logsoftmax(fused_logits_class+eps), 1) 

                loss_ae = (1+domain_entropy)*class_entropy
                loss+= 0.5*self.gamma*loss_ae;
        
            if(self.model_args['RGB']["temporal-type"]=="TRN" and self.model_args['RGB']["ablation"]["grd"]):
                fused_logits_rd = reduce(lambda x, y: x + y, logits["rd"].values())
                loss_rd = 0;
                for i in range(0,self.num_clips-1):
                    loss_rd += self.criterion_rd(fused_logits_rd[:,i,:], label_d)
                # Update the loss value, weighting it by the ratio of the batch size to the total 
                # batch size (for gradient accumulation)
                self.loss_rd.update(loss_rd)
                loss += 0.5*loss_rd
            
            
            self.loss.update(torch.mean(loss_weight * loss) / (self.total_batch / self.batch_size), self.batch_size)
        else: #target
            fused_logits_class = reduce(lambda x, y: x + y, logits["class"].values())
            fused_logits_sd = reduce(lambda x, y: x + y, logits["sd"].values())
            fused_logits_td = reduce(lambda x, y: x + y, logits["td"].values())
            loss = torch.zeros([32]).to(self.device)
            if(self.model_args['RGB']["ablation"]["gsd"]):
                self.loss_sd.add(self.criterion_sd(fused_logits_sd, label_d))
                loss += 0.5*self.loss_sd.val
            if(self.model_args['RGB']["ablation"]["gtd"]):
                self.loss_td.add(self.criterion_td(fused_logits_td, label_d))
                loss += 0.5*self.loss_td.val

                #Loss ae
                softmax = nn.Softmax(dim=1)
                logsoftmax = nn.LogSoftmax(dim=1)
                
                #domain_entropy = torch.sum(-(fused_logits_td) * torch.log(fused_logits_td+eps), 1)
                #class_entropy = torch.sum(-(fused_logits_class) * torch.log(fused_logits_class+eps), 1) 

                domain_entropy = torch.sum(-softmax(fused_logits_td) * logsoftmax(fused_logits_td+eps), 1)
                class_entropy = torch.sum(-softmax(fused_logits_class) * logsoftmax(fused_logits_class+eps), 1) 


                loss_ae = (1+domain_entropy)*class_entropy
                loss+= 0.5*self.gamma*loss_ae;
            
            if(self.model_args['RGB']["temporal-type"]=="TRN" and self.model_args['RGB']["ablation"]["grd"]):
                fused_logits_rd = reduce(lambda x, y: x + y, logits["rd"].values())
                
                for i in range(0,self.num_clips-1):
                    self.loss_rd.add(self.criterion_rd(fused_logits_rd[:,i,:], label_d))
                loss += 0.5*self.loss_rd.val
            
            
            
            self.loss.add(torch.mean(loss_weight * loss) / (self.total_batch / self.batch_size), self.batch_size)
    def compute_loss2(self,logits_source,logits_target,label_class_source,label_d_source,label_d_target,loss_weight=1):
        
        Ns = self.total_batch
        Nsut = 2*Ns

        loss = 0
        # Source 
        #   class
        fused_logits_class_source = reduce(lambda x, y: x + y, logits_source["class"].values())
        loss_class_source = self.criterion_class(fused_logits_class_source, label_class_source)
        self.loss_class.update(loss_class_source)
        loss+= self.loss_class.val 

        loss_sd =0
        if(self.model_args['RGB']["ablation"]["gsd"]):
            fused_logits_sd_source = reduce(lambda x, y: x + y, logits_source["sd"].values())
            fused_logits_sd_target = reduce(lambda x, y: x + y, logits_target["sd"].values())
            loss_sd += self.criterion_sd(fused_logits_sd_source, label_d_source) 
            loss_sd += self.criterion_sd(fused_logits_sd_target, label_d_target) 
            self.loss_sd.update(loss_sd)

            loss+= loss_sd
        loss_td = 0 
        if(self.model_args['RGB']["ablation"]["gtd"]):    
            fused_logits_td_source = reduce(lambda x, y: x + y, logits_source["td"].values())
            fused_logits_td_target = reduce(lambda x, y: x + y, logits_target["td"].values())
            
            loss_td += self.criterion_td(fused_logits_td_source,label_d_source)
            loss_td += self.criterion_td(fused_logits_td_target,label_d_target)
            self.loss_td.update(loss_td)
            
            # Loss AE
            fused_logits_class_target = reduce(lambda x, y: x + y, logits_target["class"].values())
             
            loss_ae_source = computeEntropyLoss(fused_logits_td_source,fused_logits_class_source)
            loss_ae_target = computeEntropyLoss(fused_logits_td_target,fused_logits_class_target)

            self.loss_ae.update(loss_ae_source + loss_ae_target);
            
            #Update loss           
            loss += loss_td+self.gamma *self.loss_ae.val
        if(self.model_args['RGB']["temporal-type"]=="TRN" and self.model_args['RGB']["ablation"]["grd"]):
            loss_rd = self.compute_loss_rd(logits_source,logits_target,label_d_source,label_d_target)
            self.loss_rd.update(loss_rd)

            loss+=loss_rd
        #  <----–-------–-------–-------–-------–-------–-------–-------

        self.loss.update(torch.mean(loss_weight * loss) / (self.total_batch / self.batch_size), self.batch_size)
        return
    def compute_loss3(self,logits_source,logits_target,label_class_source,label_d_source,label_d_target,loss_weight=1):
        
        Ns = self.total_batch
        Nsut = 2*Ns

        loss = 0
        # Source 
        #   class
        fused_logits_class_source = reduce(lambda x, y: x + y, logits_source["class"].values())
        loss_class_source = self.criterion_class(fused_logits_class_source, label_class_source)
        self.loss_class.update(loss_class_source)
        loss+= self.loss_class.val 

        loss_sd =0
        if(self.model_args['RGB']["ablation"]["gsd"]):
            fused_logits_sd_source = reduce(lambda x, y: x + y, logits_source["sd"].values())
            fused_logits_sd_target = reduce(lambda x, y: x + y, logits_target["sd"].values())
            loss_sd += self.criterion_sd(fused_logits_sd_source, label_d_source) 
            loss_sd += self.criterion_sd(fused_logits_sd_target, label_d_target) 
            self.loss_sd.update(loss_sd)

            loss+= 0.5*self.l_s*loss_sd
        loss_td = 0 
        if(self.model_args['RGB']["ablation"]["gtd"]):    
            fused_logits_td_source = reduce(lambda x, y: x + y, logits_source["td"].values())
            fused_logits_td_target = reduce(lambda x, y: x + y, logits_target["td"].values())
            
            loss_td += self.criterion_td(fused_logits_td_source,label_d_source)
            loss_td += self.criterion_td(fused_logits_td_target,label_d_target)
            self.loss_td.update(loss_td)
            
            # Loss AE
            fused_logits_class_target = reduce(lambda x, y: x + y, logits_target["class"].values())
             
            loss_ae_source = computeEntropyLoss(fused_logits_td_source,fused_logits_class_source)
            loss_ae_target = computeEntropyLoss(fused_logits_td_target,fused_logits_class_target)

            self.loss_ae.update(loss_ae_source + loss_ae_target);
            
            #Update loss           
            loss += 0.5*self.l_s*loss_td+0.5*self.gamma *self.loss_ae.val
        if(self.model_args['RGB']["temporal-type"]=="TRN" and self.model_args['RGB']["ablation"]["grd"]):
            loss_rd = self.compute_loss_rd(logits_source,logits_target,label_d_source,label_d_target)
            self.loss_rd.update(loss_rd)

            loss+=0.5*self.l_r*loss_rd
        #  <----–-------–-------–-------–-------–-------–-------–-------

        self.loss.update(torch.mean(loss_weight * loss) / (self.total_batch / self.batch_size), self.batch_size)
        return
    def compute_loss_td(self,logits_source,logits_target,label_d_source,label_d_target):
        fused_logits_td_source = reduce(lambda x, y: x + y, logits_source["td"].values())
        fused_logits_td_target = reduce(lambda x, y: x + y, logits_target["td"].values())
        
        loss_td += self.criterion_td(fused_logits_td_source,label_d_source)
        loss_td += self.criterion_td(fused_logits_td_target,label_d_target)
        self.loss_td.update(loss_td)
        return loss_td


            
    def compute_loss_rd(self,logits_source,logits_target,label_d_source,label_d_target):
        fused_logits_rd_source = reduce(lambda x, y: x + y, logits_source["rd"].values())
        fused_logits_rd_target = reduce(lambda x, y: x + y, logits_target["rd"].values())

        loss_rd_source = 0
        loss_rd_target = 0

        for i in range(0,self.num_clips-1):
                    loss_rd_source += self.criterion_rd(fused_logits_rd_source[:,i,:], label_d_source)
                    loss_rd_target += self.criterion_rd(fused_logits_rd_target[:,i,:], label_d_target)
        
        return loss_rd_source + loss_rd_target
    def compute_accuracy(self, logits_source: Dict[str, torch.Tensor], label: torch.Tensor):
        """Fuse the logits from different modalities and compute the classification accuracy.

        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        """
        logits = logits_source["class"]
        fused_logits = reduce(lambda x, y: x + y, logits.values())
        self.accuracy.update(fused_logits, label)

    def wandb_log(self):
        """Log the current loss and top1/top5 accuracies to wandb."""
        logs = {
            'loss verb': self.loss.val, 
            'top1-accuracy': self.accuracy.avg[1],
            'top5-accuracy': self.accuracy.avg[5]
        }

        # Log the learning rate, separately for each modality.
        for m in self.modalities:
            logs[f'lr_{m}'] = self.optimizer[m].param_groups[-1]['lr']
        wandb.log(logs)

    def reduce_learning_rate(self):
        """Perform a learning rate step."""
        for m in self.modalities:
            prev_lr = self.optimizer[m].param_groups[-1]["lr"]
            new_lr = self.optimizer[m].param_groups[-1]["lr"] / 10
            self.optimizer[m].param_groups[-1]["lr"] = new_lr

            logger.info(f"Reducing learning rate modality {m}: {prev_lr} --> {new_lr}")

    def reset_loss(self):
        """Reset the classification loss.

        This method must be called after each optimization step.
        """
        self.loss_class.reset()
        self.loss_sd.reset()
        self.loss_td.reset()
        self.loss_rd.reset()

        self.loss.reset()

    def reset_acc(self):
        """Reset the classification accuracy."""
        self.accuracy.reset()

    def step(self):
        """Perform an optimization step.

        This method performs an optimization step and resets both the loss
        and the accuracy.
        """
        super().step()
        self.reset_loss()
        self.reset_acc()

    def backward(self, retain_graph: bool = False):
        """Compute the gradients for the current value of the classification loss.

        Set retain_graph to true if you need to backpropagate multiple times over
        the same computational graph.

        Parameters
        ----------
        retain_graph : bool, optional
            whether the computational graph should be retained, by default False
        """
        self.loss.val.backward(retain_graph=retain_graph)

    def get_losses(self):
        losses = torch.zeros([4])
        losses[0] = torch.mean(self.loss_class.avg)
        if(self.model_args['RGB']['ablation']['gsd']):
            losses[1] = torch.mean(self.loss_sd.avg)
        if(self.model_args['RGB']['ablation']['gtd']):    
            losses[2] = torch.mean(self.loss_td.avg)
        if(self.model_args['RGB']['ablation']['grd']):    
            losses[3] = torch.mean(self.loss_rd.avg)
        return losses
    


def computeEntropyLoss(fused_logits_td,fused_logits_class):
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
    eps = 1e-7
    domain_entropy = torch.sum(-softmax(fused_logits_td) * logsoftmax(fused_logits_td+eps), 1)
    class_entropy = torch.sum(-softmax(fused_logits_class) * logsoftmax(fused_logits_class+eps), 1) 
    loss_ae = (1+domain_entropy)*class_entropy
    return loss_ae