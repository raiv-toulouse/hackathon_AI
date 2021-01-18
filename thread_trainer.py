#
# Note -- this training script is tweaked from the original at:
#           https://github.com/pytorch/examples/tree/master/imagenet
#
# For a step-by-step guide to transfer learning with PyTorch, see:
#           https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#
import os
import numpy as np
import shutil
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from reshape import reshape_model
from PyQt5.QtCore import QThread,pyqtSignal
from time import sleep
from torch.utils.tensorboard import SummaryWriter
import torchvision

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

LOSS_THRESHOLD = 10  # Start ploting loss if under LOSS_THRESHOLD
#
# statistic averaging
#
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} '
        return fmtstr.format(**self.__dict__)

#
# progress metering
#
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Thread_Trainer(QThread):
    signalEndTraining = pyqtSignal()
    signalAccuracyLossData = pyqtSignal(int,float,float)

    def __init__(self,model_dir,data_dir,nb_epochs,parent=None):
        super(Thread_Trainer, self).__init__(parent)
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.arch = 'resnet18'
        self.resolution = 224
        self.workers = 2
        self.epochs = nb_epochs # number of total epochs to run
        self.start_epoch = 0 # manual epoch number (useful on restarts)
        self.batch_size = 8 # mini-batch size (default: 8), this is the total batch size of all GPUs on the current node when  using Data Parallel or Distributed Data Parallel
        self.lr = 0.1 # initial learning rate
        self.momentum = 0.9 # momentum
        self.weight_decay = 1e-4 # weight decay (default: 1e-4)
        self.print_freq = 10 # print frequency (default: 10)
        self.pretrained = True # use pre-trained model
        self.world_size = -1 # number of nodes for distributed training
        self.rank = -1 # node rank for distributed training
        self.dist_url = 'tcp://224.66.41.62:23456' # url used to set up distributed training
        self.dist_backend = 'nccl' # distributed backend
        self.seed = None # seed for initializing training.
        self.gpu = 0 # GPU id to use.
        self.multiprocessing_distributed = False # ', action='store_true',  help='Use multi-processing distributed training to launch '  'N processes per node, which has N GPUs. This is the ' 'fastest way to use PyTorch for either single node or '  'multi node data parallel training')
        self.best_acc1 = 0
        # transforms
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    #
    # initiate worker threads (if using distributed multi-GPU)
    #
    def run(self):
        self.gpu = 0
        print("Use GPU: {} for training".format(self.gpu))
    
        # data loading code
        traindir = os.path.join(self.data_dir, 'train')
        valdir = os.path.join(self.data_dir, 'val')
        normalize = transforms.Normalize(mean=self.mean,
                                         std=self.std)
    
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                #transforms.Resize(224),
                transforms.RandomResizedCrop(self.resolution, scale=(0.5, 1.0), ratio=(1.0,1.0)),  # Pour ne pas avoir de déformation de l'image
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    
        num_classes = len(train_dataset.classes)
        print('=> dataset classes:  ' + str(num_classes) + ' ' + str(train_dataset.classes))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, pin_memory=True, sampler=None)

        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                normalize,
            ]))

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=True)

        val_loader_all = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=len(val_dataset), shuffle=False,
            num_workers=self.workers, pin_memory=True)
    
        # create or load the model if using pre-trained (the default)
        print("=> using pre-trained model '{}'".format(self.arch))
        model = models.__dict__[self.arch](pretrained=True)

        # reshape the model for the number of classes in the dataset
        model_cpu = reshape_model(model, self.arch, num_classes)
    
        # transfer the model to the GPU that it should be run on
        torch.cuda.set_device(self.gpu)
        model = model_cpu.cuda(self.gpu)

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(self.gpu)
    
        optimizer = torch.optim.SGD(model.parameters(), self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        cudnn.benchmark = True

        # default `log_dir` is "runs" - we'll be more specific here
        writer = SummaryWriter('runs/hackathon_AI')
        # get some random training images
        dataiter = iter(train_loader)
        images, labels = dataiter.next()

        # create grid of images
        img_grid = torchvision.utils.make_grid(images)
        # get and show the unnormalized images
        img_grid = self.show_img(img_grid)
        # write to tensorboard
        writer.add_image('hackathon', img_grid)
        #writer.add_graph(model_cpu, images)  # bug : RuntimeError: Expected object of device type cuda but got device type cpu for argument #1 'self' in call to _thnn_conv2d_forward

        # train for the specified number of epochs
        for epoch in range(self.start_epoch, self.epochs):
            print('Begin epoch #{}'.format(epoch))
            sleep(0.001)
    
            # decay the learning rate
            self.adjust_learning_rate(optimizer, epoch)
    
            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch, num_classes)
    
            # evaluate on validation set
            acc1, loss, ret_images, ret_target = self.validate(val_loader, model, criterion, num_classes)

            # save on Tensorboard
            writer.add_scalar('validation loss',loss,epoch)
            writer.add_scalar('validation accuracy',acc1,epoch)
            # ...log a Matplotlib Figure showing the model's predictions on all validation images

            dataiter_val = iter(val_loader_all)
            images_val, labels_val = dataiter.next()

            writer.add_figure('predictions vs. actuals',
                            self.plot_classes_preds(model, images_val, labels_val, train_dataset.classes),
                            global_step=epoch)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > self.best_acc1
            self.best_acc1 = max(acc1, self.best_acc1)
    
            if not self.multiprocessing_distributed or (self.multiprocessing_distributed
                    and self.rank % 1 == 0):
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.arch,
                    'resolution': self.resolution,
                    'num_classes': num_classes,
                    'state_dict': model.state_dict(),
                    'best_acc1': self.best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best)
        writer.close()
        self.signalEndTraining.emit()

    #
    # train one epoch
    #
    def train(self, train_loader, model, criterion, optimizer, epoch, num_classes):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':6.2f')
        top1 = AverageMeter('Accuracy', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1],
            prefix="Epoch: [{}]".format(epoch))
    
        # switch to train mode
        model.train()
    
        # get the start time
        epoch_start = time.time()
        end = epoch_start
    
        # train over each image batch from the dataset
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
    
            if self.gpu is not None:
                images = images.cuda(self.gpu, non_blocking=True)
            target = target.cuda(self.gpu, non_blocking=True)
    
            # compute output
            output = model(images)
            loss = criterion(output, target)
    
            # measure accuracy and record loss
            acc1 = self.accuracy(output, target)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # Send data to add in plots
            self.signalAccuracyLossData.emit(epoch,loss,acc1)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % self.print_freq == 0:
                progress.display(i)
        
        print("Epoch: [{:d}] completed, elapsed time {:6.3f} seconds".format(epoch, time.time() - epoch_start))
    
    
    #
    # measure model performance across the val dataset
    #
    def validate(self, val_loader, model, criterion, num_classes):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Accuracy', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1],
            prefix='Test: ')
    
        # switch to evaluate mode
        model.eval()
    
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                if self.gpu is not None:
                    images = images.cuda(self.gpu, non_blocking=True)
                target = target.cuda(self.gpu, non_blocking=True)
    
                # compute output
                output = model(images)
                loss = criterion(output, target)
    
                # measure accuracy and record loss
                acc1 = self.accuracy(output, target)
                display_loss = min(loss.item(), LOSS_THRESHOLD)
                losses.update(display_loss, images.size(0))
                top1.update(acc1[0], images.size(0))
    
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
    
                if i % self.print_freq == 0:
                    progress.display(i)

                if i == 0:
                    ret_images = images
                    ret_target = target

            print(' * Accuray {top1.avg:.3f} '.format(top1=top1))
    
        return top1.avg, losses.avg, ret_images, ret_target
    
    
    #
    # save model checkpoint
    #
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
        """Save a model checkpoint file, along with the best-performing model if applicable"""
    
        # if saving to an output directory, make sure it exists
        if self.model_dir:
            model_path = os.path.expanduser(self.model_dir)
    
            if not os.path.exists(model_path):
                os.mkdir(model_path)
    
            filename = os.path.join(model_path, filename)
            best_filename = os.path.join(model_path, best_filename)
    
        # save the checkpoint
        torch.save(state, filename)
    
        # earmark the best checkpoint
        if is_best:
            shutil.copyfile(filename, best_filename)
            print("saved best model to:  " + best_filename)
        else:
            print("saved checkpoint to:  " + filename)
    
    
    #
    # learning rate decay
    #
    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    
    #
    # compute the accuracy for a given result
    #
    def accuracy(self, output, target):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            batch_size = target.size(0)
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
            return correct_k.mul_(100.0 / batch_size)

    def inverse_normalize(self,tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    # to get and show proper images
    def show_img(self,img):
        # unnormalize the images
        img = self.inverse_normalize(tensor=img)
        npimg = img.numpy()
        return npimg  # return the unnormalized images

    # helper function to show an image
    # (used in the `plot_classes_preds` function below)
    def matplotlib_imshow(self, img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5     # unnormalize
        img_cpu = img.cpu()
        npimg = img_cpu.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def images_to_probs(self, net, images):
        '''
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        '''
        output = net(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        tensor_cpu = preds_tensor.cpu()
        preds = np.squeeze(tensor_cpu.numpy())
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

    def plot_classes_preds(self, net, images, labels, classes):
        '''
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        '''
        preds, probs = self.images_to_probs(net, images)
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(12, 48))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
            self.matplotlib_imshow(images[idx], one_channel=True)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]],
                probs[idx] * 100.0,
                classes[labels[idx]]),
                color=("green" if preds[idx] == labels[idx].item() else "red"))
        return fig