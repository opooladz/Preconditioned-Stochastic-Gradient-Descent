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
from torch.autograd import Variable
import random
import copy
import math
from data_loaders.loaders import *
from models.resnet import ResNet18
from reproduce.seeds import *

parser = argparse.ArgumentParser()
parser.add_argument("--experiment",                 default='cifar10',                      help="pick which experiment")
parser.add_argument("--stage2",                     default='cifar10',                      help="pick stage2 of experiment")
parser.add_argument("--epoch_concept_switch",       default=201,                            help="when should we switch to stage2 of experiment")
parser.add_argument("--num_epoch",                  default=200,                            help="how long should our full experiment be")
parser.add_argument("--device",                     default='cuda:0',                       help="for example, cuda:0")
parser.add_argument("--optimizer",                  default='PSGD_XMat',                    help="choices are SGD, PSGD_XMat and PSGD_UVd")
parser.add_argument("--lr_scheduler",               default='cos',                          help="choices are stage and cos")
parser.add_argument("--shortcut_connection",        default=1,           type=int,          help="choices are 0 and 1")
parser.add_argument('--seed',                       default=2048,        type=int,          help='random seed')
parser.add_argument('--data_seed',                  default=1738,        type=int,          help='random data_seed')
parser.add_argument('--data_root',                  default='./data/ntga_cnn_best',         help='root of data')


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
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(data_loader):
        # for batch_idx, (inputs, targets) in tqdm( enumerate(data_loader), total = len(data_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total

    return accuracy



def train(net, device, train_loader, loss_cores,noise_prior_cur):
    net.train()  # do not forget it as there is BN
    total = 0
    train_loss = 0
    correct = 0
    v_list = np.zeros(num_training_samples)
    idx_each_class_noisy = [[] for i in range(num_classes)]
    if not isinstance(noise_prior_cur, torch.Tensor):
        noise_prior_cur = torch.tensor(noise_prior_cur.astype('float32')).to(device).unsqueeze(0)

    for batch_idx, (inputs, targets, indexes) in tqdm( enumerate(train_loader), total = len(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)
        class_list = range(num_classes)

        outputs = net(inputs)

        loss, loss_v = loss_cores(epoch, outputs, targets ,class_list,ind, True, noise_prior = noise_prior_cur )
        loss + sum(
            [torch.sum(decay * torch.rand_like(param) * param * param) for param in net.parameters()]
        )
        v_list[ind] = loss_v
        for i in range(batch_size):
            if loss_v[i] == 0:
                idx_each_class_noisy[targets[i]].append(ind[i])

        def closure():
            """
            Weight decaying is explicitly realized by adding L2 regularization to the loss
            """

            return loss, loss_v , outputs

        loss, loss_v, outputs = opt.step(closure)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # make new noise prior
    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(num_classes)]
    noise_prior_delta = np.array(class_size_noisy)
    noise_prior_cur = noise_prior*num_training_samples - noise_prior_delta
    noise_prior_cur = noise_prior_cur/sum(noise_prior_cur)

    train_accuracy = 100.0 * correct / total    
    return train_loss, train_accuracy, noise_prior_cur

set_seed(args.seed)
net = ResNet18().to(device)

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
    )
else:
    # PSGD with low rank approximation preconditioner
    opt = psgd.UVd(
        net.parameters(),
        rank_of_approximation = 50,
        lr_params = lr0,
        momentum = 0.9,
        preconditioner_update_probability = 0.1,
    )

# stage 1 of experiment
# please note noisy label experiment requires different training loop -- see psgd_cifar10_noisy_label.py 
train_loader, init_noise_prior, test_loader = get_dataset(experiment, batchsize, data_root, seed, data_seed)
num_classes = 10
num_training_samples = 50000


criterion = nn.CrossEntropyLoss()

def loss_cores(epoch, y, t,class_list, ind, noise_or_not, noise_prior = None):
    beta = f_beta(epoch)

    loss = F.cross_entropy(y, t, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)
    loss_div_numpy = float(np.array(0))
    loss_ = -torch.log(F.softmax(y) + 1e-8)
    # sel metric
    loss_sel =  loss - torch.mean(loss_,1)
    if noise_prior is None:
        loss =  loss - beta*torch.mean(loss_,1)
    else:
        loss =  loss - beta*torch.sum(torch.mul(noise_prior, loss_),1)

    loss_div_numpy = loss_sel.data.cpu().numpy()
    for i in range(len(loss_numpy)):
        if epoch <=30:
            loss_v[i] = 1.0
        elif loss_div_numpy[i] <= 0:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).to(device)
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_)/100000000
    else:
        return torch.sum(loss_)/sum(loss_v), loss_v.astype(int)

#TODO: extend f_beta to 200 epochs -- needs some deeper understanding 
def f_beta(epoch):
    beta1 = np.linspace(0.0, 0.0, num=10)
    beta2 = np.linspace(0.0, 2, num=30)
    beta3 = np.linspace(2, 2, num=60)

    beta = np.concatenate((beta1,beta2,beta3),axis=0)
    return beta[epoch]

num_epoch = 100
train_accs = []
test_accs = []
noise_prior = copy.deepcopy(init_noise_prior)
noise_prior_cur = noise_prior
test_accuracy_best = -1
for epoch in range(num_epoch):
    if lr_scheduler == 'cos':
        opt.lr_params = lr0*(1 + math.cos(math.pi*epoch/num_epoch))/2
    else:
        # schedule the learning rate
        if epoch == int(num_epoch * 0.7):
            opt.lr_params *= 0.1
        if epoch == int(num_epoch * 0.9):
            opt.lr_params *= 0.1


    train_loss, train_accuracy, noise_prior_cur = train(net, device, train_loader, loss_cores, noise_prior_cur)
    test_accuracy = test(net, device, test_loader, criterion)
    if test_accuracy > test_accuracy_best:
        test_accuracy_best = test_accuracy    

    print(
        "epoch: {}; train loss: {:.2f}; train accuracy: {:.2f}; test accuracy: {:.2f}; best test accuracy: {:.2f}".format(
            epoch + 1, train_loss, train_accuracy, test_accuracy, test_accuracy_best
        )
    )

    train_accs.append(train_accuracy)
    test_accs.append(test_accuracy)
print("train_accuracy: {}".format(train_accs))
print("test_accuracy: {}".format(test_accs))


## if want a stage2...

if stage2 == 'cifar10':
    num_epoch = 100
    train_accs = []
    test_accs = []
    train_loader_clean, test_loader_clean = get_dataset('cifar10',batchsize,data_root,seed,data_seed)
    for epoch in range(num_epoch):
        if lr_scheduler == 'cos':
            opt.lr_params = lr0*(1 + math.cos(math.pi*epoch/num_epoch))/2
        else:
            # schedule the learning rate
            if epoch == int(num_epoch * 0.7):
                opt.lr_params *= 0.1
            if epoch == int(num_epoch * 0.9):
                opt.lr_params *= 0.1

        net.train()  # do not forget it as there is BN
        total = 0
        train_loss = 0
        correct = 0
        # for batch_idx, (inputs, targets) in enumerate(train_loader):
        for batch_idx, (inputs, targets) in tqdm( enumerate(train_loader_clean), total = len(train_loader_clean)):
            inputs, targets = inputs.to(device), targets.to(device)
            class_list = range(num_classes)

            outputs = net(inputs)

            loss, loss_v = loss_cores(epoch, outputs, targets ,class_list, False, noise_prior = None )
            loss + sum(
                [torch.sum(decay * torch.rand_like(param) * param * param) for param in net.parameters()]
            )

            def closure():
                """
                Weight decaying is explicitly realized by adding L2 regularization to the loss
                """
                return loss

            loss = opt.step(closure)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_accuracy = 100.0 * correct / total
        


        test_accuracy = test_clean(net, device, test_loader_clean, criterion)
        if test_accuracy > test_accuracy_best:
            test_accuracy_best = test_accuracy

        print(
            "epoch: {}; train loss: {:.2f}; train accuracy: {:.2f}; test accuracy: {:.2f}; best test accuracy: {:.2f}".format(
                epoch + 1, train_loss, train_accuracy, test_accuracy, test_accuracy_best
            )
        )

        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)
    print("train_accuracy: {}".format(train_accs))
    print("test_accuracy: {}".format(test_accs))    
