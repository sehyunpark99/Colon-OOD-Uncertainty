# Data Loader for Colonoscopy
"""
Create train, valid, test iterators for Colonoscopy
# Train AD: 1100, HP 1050 = 2150
# Val AD: 180, HP: 120 = 300
# Int.val AD: 278, HP: 94 => Deployment set
"""

import torch
import numpy as np
from torch.utils.data import Subset

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

traindata_dir = "/home/shpark/Colonoscopy_Dataset/Train(SNUH)/"
valdata_dir = "/home/shpark/Colonoscopy_Dataset/Val(SNUH)/"
testdata_dir = "/home/shpark/Colonoscopy_Dataset/Int.val(SNUH)"

def get_train_valid_loader(batch_size, augment=False, val_seed=None, val_size=0.1, num_workers=4, pin_memory=False, **kwargs):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the Colonoscopy dataset. 
    Params:
    ------
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - val_seed: fix seed for reproducibility.
    - val_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] val_size should be in the range [0, 1]."
    assert (val_size >= 0) and (val_size <= 1), error_msg

    # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)
    # Normalize 바꿔보기
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.592, 0.4634, 0.3060], std=[0.1470, 0.1530, 0.1288])


    # define transforms
    # to 128X128 as in CNN study
    val_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), normalize,])

    if augment:
        train_transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.RandomCrop(224, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,]
        )
    else:
        train_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), normalize,])

    # load the dataset
    traindata_dir = "/home/shpark/Colonoscopy_Dataset/Train(SNUH)/"
    valdata_dir = "/home/shpark/Colonoscopy_Dataset/Val(SNUH)/"
    val_dataset = datasets.ImageFolder(valdata_dir, val_transform)
    train_dataset = datasets.ImageFolder(traindata_dir, train_transform)

    # num_train = len(train_dataset) # 2150
    # indices = list(range(num_train))
    # split = int(np.floor(val_size * num_train))

    # np.random.seed(val_seed)
    # np.random.shuffle(indices)

    # train_idx, valid_idx = indices[split:], indices[:split]

    # train_subset = Subset(train_dataset, train_idx)
    # valid_subset = Subset(val_dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False,
    )

    return (train_loader, valid_loader)


def get_test_loader(batch_size, num_workers=4, pin_memory=False, **kwargs):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.592, 0.4634, 0.3060], std=[0.1470, 0.1530, 0.1288])

    # define transform
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), normalize,])
    testdata_dir = "/home/shpark/Colonoscopy_Dataset/Int.val(SNUH)"
    dataset = datasets.ImageFolder(testdata_dir, transform) 

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def get_deploy_loader(batch_size, num_workers=4, pin_memory=False, **kwargs):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.592, 0.4634, 0.3060], std=[0.1470, 0.1530, 0.1288])

    # define transform
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), normalize,])
    
    # Ewha
    ewha_dir = "/home/shpark/Colonoscopy_Dataset/Ext.val/Ewha"
    ewha_dataset = datasets.ImageFolder(ewha_dir, transform) 
    ewha_data_loader = torch.utils.data.DataLoader(
        ewha_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )
    # Yonsei
    yonsei_dir = "/home/shpark/Colonoscopy_Dataset/Ext.val/Yonsei"
    yonsei_dataset = datasets.ImageFolder(yonsei_dir, transform) 
    yonsei_data_loader = torch.utils.data.DataLoader(
        yonsei_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )
    # Asan
    asan_dir = "/home/shpark/Colonoscopy_Dataset/Ext.val/Asan"
    asan_dataset = datasets.ImageFolder(asan_dir, transform) 
    asan_data_loader = torch.utils.data.DataLoader(
        asan_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    return ewha_data_loader, yonsei_data_loader, asan_data_loader

def get_ood_test_loader(batch_size, num_workers=4, pin_memory=False, **kwargs):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.592, 0.4634, 0.3060], std=[0.1470, 0.1530, 0.1288])

    # define transform
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), normalize,])
    testdata_dir = "/home/shpark/Colonoscopy_Dataset/SSL_SP"
    dataset = datasets.ImageFolder(testdata_dir, transform) 

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
