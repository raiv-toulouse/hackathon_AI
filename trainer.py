#
# Note -- this training script is tweaked from the original at:
#           https://github.com/pytorch/examples/tree/master/imagenet
#
# For a step-by-step guide to transfer learning with PyTorch, see:
#           https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from reshape import reshape_model

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
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


class Trainer:
    def __init__(self,model_dir,data_dir):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.arch = 'resnet18'
        self.resolution = 224
        self.workers = 2
        self.epochs = 10 # number of total epochs to run
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

    
    #
    # initiate worker threads (if using distributed multi-GPU)
    #
    def main(self):   
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')
        #if self.gpu is not None:
        #    warnings.warn('You have chosen a specific GPU. This will completely '
        #                  'disable data parallelism.')
        if self.dist_url == "env://" and self.world_size == -1:
            self.world_size = int(os.environ["WORLD_SIZE"])
        self.distributed = self.world_size > 1 or self.multiprocessing_distributed
        ngpus_per_node = torch.cuda.device_count()
        # Simply call main_worker function
        self.main_worker(self.gpu, ngpus_per_node)
    
    
    #
    # worker thread (per-GPU)
    #
    def main_worker(self,gpu, ngpus_per_node):

        self.gpu = gpu
    
        if self.gpu is not None:
            print("Use GPU: {} for training".format(self.gpu))
    
        if self.distributed:
            if self.dist_url == "env://" and self.rank == -1:
                self.rank = int(os.environ["RANK"])
            if self.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.rank = self.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
    
        # data loading code
        traindir = os.path.join(self.data_dir, 'train')
        valdir = os.path.join(self.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                #transforms.Resize(224),
                transforms.RandomResizedCrop(self.resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    
        num_classes = len(train_dataset.classes)
        print('=> dataset classes:  ' + str(num_classes) + ' ' + str(train_dataset.classes))
    
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
    
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=(train_sampler is None),
            num_workers=self.workers, pin_memory=True, sampler=train_sampler)
    
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=True)
    
        # create or load the model if using pre-trained (the default)
        if self.pretrained:
            print("=> using pre-trained model '{}'".format(self.arch))
            model = models.__dict__[self.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(self.arch))
            model = models.__dict__[self.arch]()
    
        # reshape the model for the number of classes in the dataset
        model = reshape_model(model, self.arch, num_classes)
    
        # transfer the model to the GPU that it should be run on
        if self.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                model.cuda(self.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.batch_size = int(self.batch_size / ngpus_per_node)
                self.workers = int(self.workers / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif self.gpu is not None:
            torch.cuda.set_device(self.gpu)
            model = model.cuda(self.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if self.arch.startswith('alexnet') or self.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
    
        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(self.gpu)
    
        optimizer = torch.optim.SGD(model.parameters(), self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
    
    
        cudnn.benchmark = True
    
    
        # train for the specified number of epochs
        for epoch in range(self.start_epoch, self.epochs):
            if self.distributed:
                train_sampler.set_epoch(epoch)
    
            # decay the learning rate
            self.adjust_learning_rate(optimizer, epoch)
    
            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch, num_classes)
    
            # evaluate on validation set
            acc1 = self.validate(val_loader, model, criterion, num_classes)
    
            # remember best acc@1 and save checkpoint
            is_best = acc1 > self.best_acc1
            self.best_acc1 = max(acc1, self.best_acc1)
    
            if not self.multiprocessing_distributed or (self.multiprocessing_distributed
                    and self.rank % ngpus_per_node == 0):
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.arch,
                    'resolution': self.resolution,
                    'num_classes': num_classes,
                    'state_dict': model.state_dict(),
                    'best_acc1': self.best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best)
    
    
    #
    # train one epoch
    #
    def train(self, train_loader, model, criterion, optimizer, epoch, num_classes):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
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
            acc1, acc5 = self.accuracy(output, target, topk=(1, min(5, num_classes)))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
    
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
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
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
                acc1, acc5 = self.accuracy(output, target, topk=(1, min(5, num_classes)))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
    
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
    
                if i % self.print_freq == 0:
                    progress.display(i)

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
    
        return top1.avg
    
    
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
    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
    
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
    
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
