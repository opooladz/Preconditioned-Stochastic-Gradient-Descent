'''DataLoaders for PSGD Vision Experiments

Standard DataLoader: build_dataset

Imbalanced DataLoader: create_class_imb_cifar10

NTK Attacked DataLoader : AdvDataSet

Noisy Label DataLoader: input_dataset

Blurred DataLoader: build_dataset_blurred

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np

from torch.utils.data import Dataset, random_split

import os
import pandas as pd
from torchvision.io import read_image
import random
from .cifar import CIFAR10

def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Standard DataLoader

def build_train_loader(batchsize):
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True
    )
    return train_loader

def build_test_loader(batchsize):
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batchsize, shuffle=False
    )
    return test_loader

def build_dataset(batchsize):
    print("==> Preparing Standard CIFAR10 data..")
    return build_train_loader(batchsize), build_test_loader(batchsize)



# Class Imbalanced Datasets

def create_class_imb_cifar10(batchsize,data_root,seed,data_seed):
    print("==> Preparing Class Imbalanced data..")
    # torch.cuda.manual_seed(42)
    # torch.manual_seed(42)


    num_cls = 10
    classimb_ratio = 0.5
    # path = '/pytorch-cifar-master/data'
    path = data_root
    fullset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)

    set_seed(seed)
    if True:
        samples_per_class = torch.zeros(num_cls)
        for i in range(num_cls):
            samples_per_class[i] = len(torch.where(torch.Tensor(fullset.targets) == i)[0])
        min_samples = int(torch.min(samples_per_class) * 0.1)
        selected_classes = np.random.choice(np.arange(num_cls), size=int( classimb_ratio * num_cls), replace=False)
        for i in range(num_cls):
            if i == 0:
                if i in selected_classes:
                    subset_idxs = list(
                        np.random.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                            size=min_samples,
                                            replace=False))
                else:
                    subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
            else:
                if i in selected_classes:
                    batch_subset_idxs = list(
                        np.random.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                            size=min_samples,
                                            replace=False))
                else:
                    batch_subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                subset_idxs.extend(batch_subset_idxs)
        trainset = torch.utils.data.Subset(fullset, subset_idxs)

    set_seed(data_seed)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True
    )
    

    return train_loader, build_test_loader(batchsize)


# NTK Attacked Dataset

class AdvDataSet(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = np.argmax(np.load(annotations_file),axis=1)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = np.load(img_dir)
        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.dataset[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label    


def create_adv_dataset(batchsize,data_root,seed,data_seed):
    print("==> Preparing ntk attacked data..")
    x_test = data_root + 'x_train_cifar10_ntga_cnn_best.npy'
    y_test = data_root + 'y_train_cifar10.npy'

    x_train  = data_root + 'x_val_cifar10.npy'
    y_train = data_root + 'y_val_cifar10.npy'    

    set_seed(seed)
    trainset = AdvDataSet(y_train,x_train)

    testset = AdvDataSet(y_test,x_test)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True
    ) 
       
    set_seed(data_seed)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batchsize, shuffle=False
    )
    return train_loader, test_loader 

# Noisy Label Dataset


def input_dataset(noise_type, noise_ratio):
    train_dataset = CIFAR10(root='./data/',
                            download=False,  
                            train=True, 
                            transform = transform_train,
                            noise_type=noise_type,
                            noise_rate=noise_ratio
                       )
    test_dataset = CIFAR10(root='./data/',
                            download=False,  
                            train=False, 
                            transform = transform_test,
                            noise_type=noise_type,
                            noise_rate=noise_ratio
                      )
    num_classes = 10
    num_training_samples = 50000

    return train_dataset, test_dataset, num_classes, num_training_samples

def create_noisy_dataset(batchsize,seed,data_seed,soft=False):
    print("==> Preparing noisy data..")
    set_seed(seed)
    trainset, testset, num_classes , num_training_samples = input_dataset('instance',0.6)
    set_seed(data_seed)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batchsize, shuffle=False
    )
    if soft:
        return train_loader, trainset.noise_prior, test_loader#, num_classes, num_training_samples
    else:
        return train_loader, test_loader#, num_classes, num_training_samples


# Blurred CIFAR10 

transform_train_blur = transforms.Compose(
    [   transforms.Resize(8),
        transforms.Resize(32,interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        
    ]
)


def build_dataset_blurred(batchsize,seed,data_seed):
    print("==> Preparing blurred data..")

    set_seed(seed)
    trainset_blur = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train_blur
    )
    set_seed(data_seed)
    train_loader_blur = torch.utils.data.DataLoader(
        trainset_blur, batch_size=batchsize, shuffle=True
    )


    return train_loader_blur,  build_test_loader(batchsize)




def get_dataset(experiment,batchsize,data_root,seed,data_seed):
    # Note: only the noisy and blurred datasets were run with two stages: 
    # 1) noisy label or blurred image for 100 epochs
    # 2) clean data for another 100 epochs
    # the code has been written so you can run this style for any of the datasets.
    if experiment == 'cifar10':
        # standard cifar10 dataset
        train_loader, test_loader = build_dataset(batchsize)

    elif experiment == 'imb':
        # class imbalanced dataset: train is imb test is standard 
        train_loader, test_loader = create_class_imb_cifar10(batchsize,data_root,seed,data_seed)

    elif experiment == 'attacked':
        # ntk attacked dataset: train is clean test is attacked
        train_loader, test_loader = create_adv_dataset(batchsize,data_root,seed,data_seed)

    elif experiment == 'noisy':
        # this just returns a noisy label dataset -- no noise prior  
        train_loader, test_loader = create_noisy_dataset(batchsize,seed,data_seed)

    elif experiment == 'blurred':
        # blur train data initially and then unblur at epoch_concept_switch this is a neuro-plasticity test
        # Observations:
        # 1) PSGD remained much more neuro plastic compared to SGD.
        # 2) Even after 100 epochs of blur PSGD can recover test accuracy of 93.5% -- a 2% decrease compared to a ~11% decrease for SGD
        train_loader, test_loader = build_dataset_blurred(batchsize,seed,data_seed)
    
    elif experiment == 'noisy_soft':
        # this comes with noise prior used in sofr training
        # noisy label dataset: train label is noisy test is clean
        # Observations:
        # 1) giffted kid syndrom with SGD but not with PSGD 
        train_loader, noise_prior, test_loader = create_noisy_dataset(batchsize,seed,data_seed,soft=True)
        return train_loader, noise_prior ,test_loader
        
    else:
        #TODO make system exit 
        print('This experiment is not supported')
    return train_loader,test_loader