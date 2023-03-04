import sys
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import preconditioned_stochastic_gradient_descent as psgd
from tqdm import tqdm
import random
import os
from data_loaders.loaders import *
from models.resnet import ResNet18
from reproduce.seeds import *

parser = argparse.ArgumentParser()
parser.add_argument("--experiment",                 default='cifar10',                      help="pick which experiment")
parser.add_argument("--stage2",                     default='cifar10',                      help="pick stage2 of experiment")
parser.add_argument("--epoch_concept_switch",       default=201,                            help="when should we switch to stage2 of experiment")
parser.add_argument("--num_epoch",                  default=200,                            help="how long should our full experiment be")
parser.add_argument("--num_runs",                   default=5,                              help="how many runs")
parser.add_argument("--device",                     default='cpu',                          help="for example, cuda:0")
parser.add_argument("--optimizer",                  default='PSGD_XMat',                    help="choices are SGD, PSGD_XMat and PSGD_UVd")
parser.add_argument("--lr_scheduler",               default='cos',                          help="choices are stage and cos")
parser.add_argument("--shortcut_connection",        default=1,           type=int,          help="choices are 0 and 1")
parser.add_argument('--seed',                       default=2048,        type=int,          help='random seed')
parser.add_argument('--data_seed',                  default=1738,        type=int,          help='random data_seed')
parser.add_argument('--data_root',                  default='./data/ntga_cnn_best/',        help='root of data')


data_see_l = [141,1024,2048,1738,1776]
seed_l [141,1024,2048,1738,1776]
args = parser.parse_args()
experiment = args.experiment
stage2 = args.stage2
num_epoch = args.num_epoch
epoch_concept_switch = args.epoch_concept_switch
device = torch.device(args.device)
optimizer = args.optimizer
lr_scheduler = args.lr_scheduler
shortcut_connection = bool(args.shortcut_connection)
seed = args.seed
data_seed = args.data_seed
data_root = args.data_root
print("Experiment Stage 1: \t\t\t{}".format(experiment))
print("Experiment Stage 2: \t\t\t{}".format(stage2))
print("Total Epochs: \t\t\t{}".format(num_epoch))
print("Change Experiment at Epoch: \t\t\t{}".format(epoch_concept_switch))
print("Device: \t\t\t{}".format(device))
print("Optimizer: \t\t\t{}".format(optimizer))
print("Learning rate schedular:\t{}".format(lr_scheduler))
print("With short connections: \t{}".format(shortcut_connection))
print("Seed: \t\t\t\t{}".format(seed))
print("Data Seed: \t\t\t{}".format(data_seed))
print("Data Root: \t\t\t{}".format(data_root))

set_seed(args.seed)
set_cuda(deterministic=True)

if optimizer == 'SGD':
    lr0 = 1.0   # 0.1 -> 1.0 when momentum factor = 0.9 as momentum in PSGD is the moving average of gradient
    decay = 5e-4
else: # PSGD_XMat or PSGD_UVd
    lr0 = 2e-2
    if shortcut_connection:
        decay = 2e-2
    else:
        decay = 1e-2

if shortcut_connection:
    batchsize = 128
else:
    batchsize = 64

def test(net, device, data_loader, criterion):
    if torch.__version__.startswith('2'):
        net = torch.compile(net)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total

    return accuracy

def train(net, device, data_loader, criterion):
    if torch.__version__.startswith('2'):
        net = torch.compile(net)
    net.train()  # do not forget it as there is BN
    total = 0
    train_loss = 0
    correct = 0
    for batch_idx, (inputs, targets) in tqdm( enumerate(data_loader), total = len(data_loader)):
        inputs, targets = inputs.to(device), targets.to(device)

        def closure():
            """
            Weight decaying is explicitly realized by adding L2 regularization to the loss
            """
            outputs = net(inputs)
            loss = criterion(outputs, targets) + sum(
                [torch.sum(decay * torch.rand_like(param) * param * param) for param in net.parameters()]
            )
            return [loss, outputs]

        loss, outputs = opt.step(closure)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_accuracy = 100.0 * correct / total    
    return train_loss, train_accuracy
train_accs_l = []
test_accs_l = []
for run in runs:
    seed_l[run]
    set_seed(seed_l[run])
    net = ResNet18(shortcut_connection=True).to(device)

    if optimizer == 'SGD':
        # SGD baseline
        opt = psgd.XMat(
            net.parameters(),
            lr_params = lr0, # note that momentum in PSGD is the moving average of gradient
            momentum = 0.9,  # so lr 0.1 becomes 1 when momentum factor is 0.9
            preconditioner_update_probability = 0.0, # PSGD reduces to SGD when P = eye()
        )
    elif optimizer == 'PSGD_XMat':
        # PSGD with X-shape matrix preconditioner
        opt = psgd.XMat(
            net.parameters(),
            lr_params = lr0,
            momentum = 0.9,
            preconditioner_update_probability = 0.1,
            exact_hessian_vector_product = False

        )
    else:
        # PSGD with low rank approximation preconditioner
        opt = psgd.UVd(
            net.parameters(),
            lr_params = lr0,
            momentum = 0.9,
            preconditioner_update_probability = 0.1,
        )

    # stage 1 of experiment
    # please note noisy label experiment requires different training loop -- see psgd_cifar10_noisy_label.py 
    train_loader, test_loader = get_dataset(experiment, batchsize, data_root, seed_l[run], data_seed_l[run])


    criterion = nn.CrossEntropyLoss()
    num_epoch = 200
    train_accs = []
    test_accs = []
    for epoch in range(num_epoch):
        if lr_scheduler == 'cos':
            opt.lr_params = lr0*(1 + math.cos(math.pi*epoch/num_epoch))/2
        else:
            # schedule the learning rate
            if epoch == int(num_epoch * 0.7):
                opt.lr_params *= 0.1
            if epoch == int(num_epoch * 0.9):
                opt.lr_params *= 0.1

        if epoch ==  epoch_concept_switch:
            # if there is a second stage of the experiment
            # note noisy label dataset requires different train loop -- see psgd_cifar10_noisy_label.py 
            
            train_loader, test_loader = get_dataset(experiment, batchsize, data_root, seed_l[run], data_seed_l[run])
        train_loss, train_accuracy = train(net, device, train_loader, criterion)
        test_accuracy = test(net, device, test_loader, criterion)
        print(
            "run: {}; epoch: {}; train loss: {:.2f}; train accuracy: {:.2f}; test accuracy: {:.2f}".format(
                run + 1
                epoch + 1, train_loss, train_accuracy, test_accuracy
            )
        )

        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)
    print("train_accuracy: {}".format(train_accs))
    print("test_accuracy: {}".format(test_accs))
    train_accs_l.append(train_accs)
    test_accs_l.append(test_accs)
print("train_accuracy: {}".format(train_accs_l))
print("test_accuracy: {}".format(test_accs_l))    
