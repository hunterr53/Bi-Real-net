import os
import sys
import shutil
from matplotlib import pyplot as plt
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torchvision

#sys.path.append("../")
from utils import *
from torchvision import datasets, transforms
from torchsummary import summary
from torch.autograd import Variable
from birealnet import birealnet18


from mnist import MNIST

# Seed
random.seed(10)

parser = argparse.ArgumentParser("birealnet")
parser.add_argument('--batch_size', type=int, default=164, help='batch size')
parser.add_argument('--epochs', type=int, default=256, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()

# CLASSES = 1000
CLASSES = 10
isCuda = True

# if not os.path.exists('log'):
#     os.mkdir('log')

# log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#     format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join('log/log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='pytorch_implementation/BiReal18_34/log/log.txt', filemode='w', 
                    level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

def main():
    if not torch.cuda.is_available():
        # sys.exit(1)
        logging.warning('No CUDA available')
        isCuda = False
    else:
        logging.info('Using CUDA')
        isCuda = True
    start_t = time.time()

    cudnn.benchmark = True if isCuda else False
    cudnn.enabled=True if isCuda else False
    logging.info("args = %s", args)

    # load model
    model = birealnet18()
    logging.info(model)
    model = nn.DataParallel(model).cuda() if isCuda else nn.DataParallel(model).cpu()

    # Get model summary
    summary(model, (3, 224, 224)) # 3x224x224

    criterion = nn.CrossEntropyLoss() # Computes the cross entropy loss between input logits and target.
    criterion = criterion.cuda() if isCuda else criterion.cpu()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth) # Restrains largest logit from getting to large at softmax
    criterion_smooth = criterion_smooth.cuda() if isCuda else criterion_smooth.cpu()

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or pname=='classifier.0.weight' or pname == 'classifier.0.bias':
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=args.learning_rate,)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
    start_epoch = 0
    best_top1_acc= 0

    checkpoint_tar = os.path.join(args.save, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_tar):
        logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        start_epoch = checkpoint['epoch']
        best_top1_acc = checkpoint['best_top1_acc']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))

    # adjust the learning rate according to the checkpoint
    for epoch in range(start_epoch):
        scheduler.step()

    # load training data
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])\
    
    cifar10_dataset = torchvision.datasets.CIFAR10(
        root="./Datasets",
        download=True, 
        train=True, 
        transform=transforms.ToTensor())

    # data augmentation
    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        Lighting(lighting_param),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])
    
    # Cifar10 dataloader
    cifar10_dataset.transform = train_transforms
    train_dataset = cifar10_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_dataset = torchvision.datasets.CIFAR10(
        root="./Datasets",
        download=True, 
        train=False, 
        transform=val_transforms)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    test_datasetBatch = unpickle('Datasets/cifar-10-batches-py/test_batch')
    # Function to show an image
    def imshow(img):
        redPixels = img[0:1024].reshape(32, 32)
        greenPixels = img[1024:2048].reshape(32, 32)
        bluePixels = img[2048:3072].reshape(32, 32)
        rgb = np.dstack((redPixels,greenPixels,bluePixels))

        plt.imshow(rgb)
        plt.show()

    for i in range(5):
        label = test_datasetBatch.get(b"labels")[i]
        img = test_datasetBatch.get(b'data')[i]
        imshow(img)
        print(label)

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        saveWeights(model, isCuda)
        train_obj, train_top1_acc,  train_top5_acc = train(epoch,  train_loader, model, criterion_smooth, optimizer, scheduler)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.save)

        epoch += 1

    training_time = (time.time() - start_t) / 36000
    print('total training time = {} hours'.format(training_time))


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    scheduler.step()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda() if isCuda else images.cpu()
        target = target.cuda() if isCuda else target.cpu()

        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)

    return losses.avg, top1.avg, top5.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda() if isCuda else images.cpu()
            target = target.cuda() if isCuda else target.cpu()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)

        print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def saveWeights(net, isCuda):    
    net = net.cpu()
    #Save weights to CSV file
    import pandas as pd
    outputLayer = False
    firstNode = True
    firstNodeWeights = ''
    stateDict = net.state_dict()

    print("\nModel's named_parameters:")
    for name, param in net.named_parameters():
        print(name, "\t", param.size())

    print("\nWrite Model Weights/Bias to file:")
    with open('pytorch_implementation\BiReal18_34\savedWeights\BiRealNetPreTrainedWeights.txt', "w+") as output:
        bnDebugCounter = 0
        downsampleDebugCounter = 0
        fcDebugCounter = 0
        for name, param in net.named_parameters():
            # break #debug
            print(name, "\t", param.size())
            weights = param.data.numpy()
            # print(weights)
            # output.write(name + "\n" + str(weights) + "\n")
            output.write(name + "\n")
            # if "bias" in name:
            #     print("bias size:", weights.size)
            #     for weight in weights:
            #         output.write(str(weight) + " ")
            #     output.write("\n")

            if "binary_conv" in name:
                print("conv size:", weights.size)
                for weight in weights: # (x, 1)
                    for weight2 in weight:
                        output.write(str(weight2) + " ")
                output.write("\n")

            elif "conv1" in name:
                print("conv size:", weights.size)
                for weight in weights: # (x, 1, 1, 1)
                    for weight2 in weight: # (1, x, 1, 1)
                        for weight3 in weight2: # (1, 1, x, 1)
                            for weight4 in weight3: # (1, 1, 1, x)
                                output.write(str(weight4) + " ")
                        # output.write(str(weight2) + " ")
                output.write("\n")
            elif "fc" in name: # FFN
                fcDebugCounter += 1
                if "bias" in name:
                    print("bias size:", weights.size)
                    for weight in weights:
                        output.write(str(weight) + " ")
                    output.write("\n")
                else:
                    print("FFN size:", weights.size)
                    print("FFN shape:", weights.shape)
                    if(outputLayer):
                        numNeurons = 10
                    else:
                        numNeurons = 1000
                        outputLayer = True

                    for weight in weights:
                        for weight2 in weight:
                            output.write(str(weight2) + " ")
                            if(firstNode):
                                # print(weight2)
                                firstNodeWeights = firstNodeWeights + (str(weight2) + "\n")

                        if(firstNode):
                                firstNode = False
                            
                    output.write("\n")
            elif "bn" in name:
                bnDebugCounter += 1
                if "bias" in name:
                    print("bias size:", weights.size)
                    for weight in weights:
                        output.write(str(weight) + " ")
                    output.write("\n")
                else:
                    print("BN weight:", weights.size)
                    for weight in weights:
                        output.write(str(weight) + " ")
                    output.write("\n")

            elif "downsample" in name:
                downsampleDebugCounter += 1
                print("Downsample weight:", weights.size)
                for weight in weights: # (x, 1, 1, 1)
                    if(len(weights.shape) == 4):
                        for weight2 in weight: # (1, x, 1, 1)
                            for weight3 in weight2: # (1, 1, x, 1)
                                for weight4 in weight3: # (1, 1, 1, x)
                                    output.write(str(weight4) + " ")
                    else: # batch norm
                        output.write(str(weight) + " ")
                output.write("\n")

            else:
                print("Unknown layer", name)
                exit(1)

        print("BN Debug Counter:", bnDebugCounter)
        print("Downsample Debug Counter:", downsampleDebugCounter)
        print("FC Debug Counter:", fcDebugCounter)

    print("\nWrite Model BN Mean/Var to file:")
    with open('pytorch_implementation\BiReal18_34\savedWeights\BiRealNetPreTrainedBN.txt', "w+") as output:
        # print("Input Running Mean", net.module.bn1.running_mean.shape)
        output.write("bn1.running_mean\n")
        for mean in net.module.bn1.running_mean:
            output.write(str(mean.numpy()) + " ")
        output.write("\n")
        # print("Input Running Var", net.module.bn1.running_var.shape)
        output.write("bn1.running_var\n")
        for var in net.module.bn1.running_var:
            output.write(str(var.numpy()) + " ")
        output.write("\n")
        
        for i in range(0, 4): # Layer 1
            # print("BN layer1." +  str(i) + ".bn1.Running Mean")
            output.write("net.module.layer1." + str(i) + ".bn1.running_mean\n")
            for mean in net.module.layer1[i].bn1.running_mean:
                output.write(str(mean.numpy()) + " ")
            output.write("\n")
            # print("BN layer1." + str(i) + ".bn1.Running Var")
            output.write("net.module.layer1." + str(i) + ".bn1.running_var\n")
            for var in net.module.layer1[i].bn1.running_var:
                output.write(str(var.numpy()) + " ")
            output.write("\n")

        for i in range(0, 4): # Layer 2
            # print("BN layer2." +  str(i) + ".bn1.Running Mean")
            output.write("net.module.layer2." + str(i) + ".bn1.running_mean\n")
            for mean in net.module.layer2[i].bn1.running_mean:
                output.write(str(mean.numpy()) + " ")
            output.write("\n")
            # print("BN layer2." + str(i) + ".bn1.Running Var")
            output.write("net.module.layer2." + str(i) + ".bn1.running_var\n")
            for var in net.module.layer2[i].bn1.running_var:
                output.write(str(var.numpy()) + " ")
            output.write("\n")
            if i == 0: # Get downsample
                # print("Downsample layer2." +  str(i) + ".downsample.module2.Running Mean")
                output.write("net.module.layer2." + str(i) + ".downsample2.running_mean\n")
                for mean in net.module.layer2[i].downsample[2].running_mean:
                    output.write(str(mean.numpy()) + " ")
                output.write("\n")
                # print("BN layer2." + str(i) + ".downsample.module2.Running Var")
                output.write("net.module.layer2." + str(i) + ".downsample2.running_var\n")
                for var in net.module.layer2[i].downsample[2].running_var:
                    output.write(str(var.numpy()) + " ")
                output.write("\n")

        for i in range(0, 4): # Layer 3
            # print("BN layer3." +  str(i) + ".bn1.Running Mean")
            output.write("net.module.layer3." + str(i) + ".bn1.running_mean\n")
            for mean in net.module.layer3[i].bn1.running_mean:
                output.write(str(mean.numpy()) + " ")
            output.write("\n")
            # print("BN layer3." + str(i) + ".bn1.Running Var")
            output.write("net.module.layer3." + str(i) + ".bn1.running_var\n")
            for var in net.module.layer3[i].bn1.running_var:
                output.write(str(var.numpy()) + " ")
            output.write("\n")
            if i == 0: # Get downsample
                # print("Downsample layer3." +  str(i) + ".downsample.module2.Running Mean")
                output.write("net.module.layer3." + str(i) + ".downsample2.running_mean\n")
                for mean in net.module.layer3[i].downsample[2].running_mean:
                    output.write(str(mean.numpy()) + " ")
                output.write("\n")
                # print("BN layer3." + str(i) + ".downsample.module2.Running Var")
                output.write("net.module.layer3." + str(i) + ".downsample2.running_var\n")
                for var in net.module.layer3[i].downsample[2].running_var:
                    output.write(str(var.numpy()) + " ")
                output.write("\n")
            
        for i in range(0, 4): # Layer 4
            # print("BN layer4." +  str(i) + ".bn1.Running Mean")
            output.write("net.module.layer4." + str(i) + ".bn1.running_mean\n")
            for mean in net.module.layer4[i].bn1.running_mean:
                output.write(str(mean.numpy()) + " ")
            output.write("\n")
            # print("BN layer4." + str(i) + ".bn1.Running Var")
            output.write("net.module.layer4." + str(i) + ".bn1.running_var\n")
            for var in net.module.layer4[i].bn1.running_var:
                output.write(str(var.numpy()) + " ")
            output.write("\n")
            if i == 0: # Get downsample
                # print("Downsample layer4." +  str(i) + ".downsample.module2.Running Mean")
                output.write("net.module.layer4." + str(i) + ".downsample2.running_mean\n")
                for mean in net.module.layer4[i].downsample[2].running_mean:
                    output.write(str(mean.numpy()) + " ")
                output.write("\n")
                # print("BN layer4." + str(i) + ".downsample.module2.Running Var")
                output.write("net.module.layer4." + str(i) + ".downsample2.running_var\n")
                for var in net.module.layer4[i].downsample[2].running_var:
                    output.write(str(var.numpy()) + " ")
                output.write("\n")

    net = net.cuda() if isCuda else net.cpu()

if __name__ == '__main__':
    main()
