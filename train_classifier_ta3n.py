from datetime import datetime
from statistics import mean
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import EpicKitchensDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import tasks
import wandb
from torch import autograd

# global variables among training functions
training_iterations = 0
modalities = None
np.random.seed(13696641)
torch.manual_seed(13696641)


def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used and wandb
    """
    logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    # this is needed for multi-GPUs systems where you just want to use a predefined set of GPUs
    #if args.gpus is not None:
    #    logger.debug('Using only these GPUs: {}'.format(args.gpus))
    #    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    # wanbd logging configuration
    if args.wandb_name is not None:
        wandb.init(group=args.wandb_name)
        #wandb.run.name = args.name + "_" + args.shift.split("-")[0] + "_" + args.shift.split("-")[-1]


def main():
    global training_iterations, modalities
    init_operations()
    modalities = args.modality

    # recover valid paths, domains, classes
    # this will output the domain conversion (D1 -> 8, et cetera) and the label list
    num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
    # device where everything is run
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(args.gpus == "mps"):
        device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
    elif(args.gpus == "cuda"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else: 
        device = torch.device("cpu")
    # these dictionaries are for more multi-modal training/testing, each key is a modality used
    models = {}
    logger.info("Instantiating models per modality")
    for m in modalities:
        logger.info('{} Net\tModality: {}'.format(args.models[m].model, m))
        # notice that here, the first parameter passed is the input dimension
        # In our case it represents the feature dimensionality which is equivalent to 1024 for I3D
        models[m] = getattr(model_list, args.models[m].model)(num_classes,[5,1024],args.models['RGB']["temporal-type"],args.models['RGB']["ablation"],device)

    # the models are wrapped into the ActionRecognition task which manages all the training steps
    action_classifier = tasks.TA3N_task("action-classifier", models, args.batch_size,
                                                args.total_batch, args.models_dir, num_classes,
                                                args.train.num_clips, args.models, args=args,device=device)
    action_classifier.load_on_gpu(device)

    if args.action == "train":
        # resume_from argument is adopted in case of restoring from a checkpoint
        if args.resume_from is not None:
            action_classifier.load_last_model(args.resume_from)
        # define number of iterations I'll do with the actual batch: we do not reason with epochs but with iterations
        # i.e. number of batches passed
        # notice, here it is multiplied by tot_batch/batch_size since gradient accumulation technique is adopted
        training_iterations = args.train.num_iter * (args.total_batch // args.batch_size)
        # all dataloaders are generated here
        train_loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset, None, None, None,
                                                                       None, load_feat=True),
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True,persistent_workers=args.dataset.persistentWorkers)
        target_loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                       'domainAdapt', args.dataset, None, None, None,
                                                                       None, load_feat=True),
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True,persistent_workers=args.dataset.persistentWorkers)

        val_loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                     'val', args.dataset, None, None, None,
                                                                     None, load_feat=True),
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.dataset.workers, pin_memory=True, drop_last=False,persistent_workers=args.dataset.persistentWorkers)
        loss_train = train(action_classifier, train_loader,target_loader, val_loader, device, num_classes)
        #                 loss_train_D1_to_D2_TRN/base_gsd1/0_gtd0/1_grd0/1_lr_sgdMomval_weightDecay
        loss_file_name = "train_images/loss_train_"+args.dataset.shift.split("-")[0]+"_to_"+args.dataset.shift.split("-")[-1]+"_" \
                            +args.models.RGB["temporal-type"]+"_gsd_"+str(args.models.RGB.ablation["gsd"])+"_gtd_"+str(args.models.RGB.ablation["gtd"]) \
                            +"_grd_"+str(args.models.RGB.ablation["grd"])+"_lr_"+str(args.models.RGB.lr)+"_sgdMom_"+str(args.models.RGB.sgd_momentum)+ \
                                "_weightDecay_"+str(args.models.RGB.weight_decay) +".pt"
        torch.save(loss_train, loss_file_name)
        
    elif args.action == "validate":
        if args.resume_from is not None:
            action_classifier.load_last_model(args.resume_from)
        val_loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                     'val', args.dataset, None, None, None,
                                                                     None, load_feat=True),
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.dataset.workers, pin_memory=True, drop_last=False)

        validate(action_classifier, val_loader, device, action_classifier.current_iter, num_classes)


def train(action_classifier, train_loader, target_loader,val_loader, device, num_classes):
    """
    function to train the model on the test set
    action_classifier: Task containing the model to be trained
    train_loader: dataloader containing the training data
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    num_classes: int, number of classes in the classification problem
    """
    global training_iterations, modalities

    data_loader_source = iter(train_loader)
    data_loader_target = iter(target_loader)
    action_classifier.train(True)
    action_classifier.zero_grad()
    iteration = action_classifier.current_iter * (args.total_batch // args.batch_size)

    loss_train = torch.zeros([int(training_iterations / (args.total_batch // args.batch_size)),4]) #
    # the batch size should be total_batch but batch accumulation is done with batch size = batch_size.
    # real_iter is the number of iterations if the batch size was really total_batch
    for i in range(int(iteration), training_iterations):
        # iteration w.r.t. the paper (w.r.t the bs to simulate).... i is the iteration with the actual bs( < tot_bs)
        real_iter = (i + 1) / (args.total_batch // args.batch_size)
        if real_iter == args.train.lr_steps:
            # learning rate decay at iteration = lr_steps
            action_classifier.reduce_learning_rate()
        # gradient_accumulation_step is a bool used to understand if we accumulated at least total_batch
        # samples' gradient
        gradient_accumulation_step = real_iter.is_integer()

        """
        Retrieve the data from the loaders
        """
        start_t = datetime.now()
        # the following code is necessary as we do not reason in epochs so as soon as the dataloader is finished we need
        # to redefine the iterator
        try:
            source_data, source_label = next(data_loader_source)
            source_label_domain = 0*torch.ones([args.batch_size], dtype=int)
    
            if (i%(len(data_loader_source._dataset)//args.batch_size))==0:
                #check if last batch of the smallest dataloader is smaller than batch_size
                raise StopIteration
        except StopIteration:
            #reset source dataloader
            data_loader_source = iter(train_loader) # TODO forse conviene separare i controlli sulla fine dei dataloader?
            source_data, source_label = next(data_loader_source)
            source_label_domain = 0*torch.ones([args.batch_size], dtype=int)

        try:
            target_data, target_label = next(data_loader_target)
            target_label_domain = 1*torch.ones([args.batch_size], dtype = int)
            
            if (i%(len(data_loader_target._dataset)//args.batch_size))==0:
                #check if last batch of the smallest dataloader is smaller than batch_size
                raise StopIteration
        except StopIteration:
            #reset target dataloader
            data_loader_target = iter(val_loader)
            target_data, target_label = next(data_loader_target)
            target_label_domain = 1*torch.ones([args.batch_size], dtype = int)
        end_t = datetime.now()

        #TODO uncomment
        #logger.info(f"Iteration {i}/{training_iterations} batch retrieved! Elapsed time = "
        #            f"{(end_t - start_t).total_seconds() // 60} m {(end_t - start_t).total_seconds() % 60} s")

        ''' Action recognition'''
        "******** We start by using the source ****************"
        source_label = source_label.to(device)
        source_label_domain= source_label_domain.to(device)
        target_label_domain= target_label_domain.to(device)
        data_s = {}
        data_t ={}
       
        for m in modalities:
            data_s[m] = source_data[m].to(device)

        for m in modalities:
            data_t[m] = target_data[m].to(device)
        

       
        #forward on source
        logits_s = action_classifier.forward(data_s)
        #forward on target
        logits_t = action_classifier.forward(data_t)
        #compute loss on source
        action_classifier.compute_loss3(logits_s,logits_t, source_label, source_label_domain,target_label_domain, loss_weight=1)
        #backward based on updated losses
        action_classifier.backward()
        #accuracy update
        action_classifier.compute_accuracy(logits_s, source_label)
        # update weights and zero gradients if total_batch samples are passed
        if gradient_accumulation_step:
            logger.info("[%d/%d]\tlast Verb loss: %.4f\tMean verb loss: %.4f\tAcc@1: %.2f%%\tAccMean@1: %.2f%%" %
                        (real_iter, args.train.num_iter, action_classifier.loss.val, action_classifier.loss.avg,
                         action_classifier.accuracy.val[1], action_classifier.accuracy.avg[1]))
            #save loss
            loss_train[i//4] = action_classifier.get_losses()
            #wandb
            wandb.log({"loss":  action_classifier.loss.val, 
                       "loss_sd": torch.mean(torch.tensor(action_classifier.loss_sd.val,dtype=float)),
                       "loss_td": torch.mean(torch.tensor(action_classifier.loss_td.val,dtype=float)),
                       "loss_rd": torch.mean(torch.tensor(action_classifier.loss_rd.val,dtype=float)),
                       "loss_ae": torch.mean(torch.tensor(action_classifier.loss_ae.val,dtype=float))
                       })

            action_classifier.check_grad()
            action_classifier.step()
            action_classifier.zero_grad()

        # every eval_freq "real iteration" (iterations on total_batch) the validation is done, notice we validate and
        # save the last 9 models
        if gradient_accumulation_step and real_iter % args.train.eval_freq == 0:
            val_metrics = validate(action_classifier, val_loader, device, int(real_iter), num_classes)
            
            if val_metrics['top1'] <= action_classifier.best_iter_score:
                logger.info("New best accuracy {:.2f}%"
                            .format(action_classifier.best_iter_score))
            else:
                logger.info("New best accuracy {:.2f}%".format(val_metrics['top1']))
                action_classifier.best_iter = real_iter
                action_classifier.best_iter_score = val_metrics['top1']

            wandb.log({"acc":  action_classifier.best_iter_score})
            action_classifier.save_model(real_iter, val_metrics['top1'], prefix=None)
            action_classifier.train(True)

    return loss_train

def validate(model, val_loader, device, it, num_classes):
    """
    function to validate the model on the test set
    model: Task containing the model to be tested
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    num_classes: int, number of classes in the classification problem
    """
    global modalities

    model.reset_acc()
    model.train(False)
    logits = {}

    # Iterate over the models
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader):
            label = label.to(device)
            logits = model(data)
            model.compute_accuracy(logits, label)

            if (i_val + 1) % (len(val_loader) // 5) == 0:
                logger.info("[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(i_val + 1, len(val_loader),
                                                                          model.accuracy.avg[1], model.accuracy.avg[5]))

        class_accuracies = [(x / y) * 100 for x, y in zip(model.accuracy.correct, model.accuracy.total)]
        logger.info('Final accuracy: top1 = %.2f%%\ttop5 = %.2f%%' % (model.accuracy.avg[1],
                                                                      model.accuracy.avg[5]))
        for i_class, class_acc in enumerate(class_accuracies):
            logger.info('Class %d = [%d/%d] = %.2f%%' % (i_class,
                                                         int(model.accuracy.correct[i_class]),
                                                         int(model.accuracy.total[i_class]),
                                                         class_acc))

    logger.info('Accuracy by averaging class accuracies (same weight for each class): {}%'
                .format(np.array(class_accuracies).mean(axis=0)))
    test_results = {'top1': model.accuracy.avg[1], 'top5': model.accuracy.avg[5],
                    'class_accuracies': np.array(class_accuracies)}

   
    with open(os.path.join(args.log_dir, f'val_precision_{args.dataset.shift.split("-")[0]}-'
                                         f'{args.dataset.shift.split("-")[-1]}.txt'), 'a+') as f:
        f.write("[%d/%d]\tAcc@top1: %.2f%%\n" % (it, args.train.num_iter, test_results['top1']))

    return test_results


if __name__ == '__main__':
    main()