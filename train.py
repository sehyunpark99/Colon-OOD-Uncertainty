"""
Script for training a single model for OOD detection.
"""
import time
import json
import torch
import torch.nn as nn
import argparse
from torch import optim
import torch.backends.cudnn as cudnn

# Import dataloaders
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn
import data.dirty_mnist as dirty_mnist
from data.imagenet import imagenet_dataloader
import colon_data

# Import network models
from net.lenet import lenet
from net.resnet import resnet18, resnet50
from net.wide_resnet import wrn
from net.vgg import vgg16

# Import train and validation utilities
from utils.args import training_args
from utils.eval_utils import get_eval_stats
from utils.train_utils import model_save_name
from utils.train_utils import train_single_epoch, test_single_epoch

# Tensorboard utilities
from torch.utils.tensorboard import SummaryWriter

dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10, "dirty_mnist": 10, "colonoscopy": 2, "imagenet": 1000}

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "dirty_mnist": dirty_mnist,
    "colonoscopy": colon_data,
}

models = {
    "lenet": lenet,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "wide_resnet": wrn,
    "vgg16": vgg16,
}

if __name__ == "__main__":

    args = training_args().parse_args()

    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)

    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    net = models[args.model](
        spectral_normalization=args.sn,
        mod=args.mod,
        coeff=args.coeff,
        num_classes=2,
        mnist="mnist" in args.dataset,
    )
    # Checkpoint
    # ckpt = torch.load(str('/home/shpark/DDU/resnet50_sn_3.0_1_50_modified.pth'))
    # net.load_state_dict(ckpt, strict=False)
    # net.fc = nn.Linear(net.fc.in_features, 2)

    if args.gpu:
        net.cuda()
        device_ids = [0, 1, 2, 3]  # Change these IDs according to your setup
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        cudnn.benchmark = True

    opt_params = net.parameters()
    if args.optimiser == "sgd":
        optimizer = optim.SGD(
            opt_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimiser == "adam":
        optimizer = optim.Adam(opt_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1
    )

    if args.dataset == "imagenet":
        print("Data: ImageNet")
        train_loader, _ = imagenet_dataloader(root=args.dataset_root,
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            val_size=0.1,
            val_seed=args.seed,
            pin_memory=args.gpu,
        )
    else:
        train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            root=args.dataset_root,
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            val_size=0.1,
            val_seed=args.seed,
            pin_memory=args.gpu,
        )

    # Creating summary writer in tensorboard
    curr_time = time.strftime("%Y-%m-%d_%H:%M:%S")
    log_dir = args.save_loc + args.dataset + "stats_logging_" + curr_time + "/"
    writer = SummaryWriter(log_dir)
    print("Training started at:", curr_time)

    training_set_loss = {}
    training_set_accuracy = {}
    test_set_loss = {}
    test_set_accuracy = {}

    save_name = model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)
    save_name = save_name + "_" + "mean_0.592"
    print("Model save name", save_name)
    start_time = time.time()

    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(0, args.epoch):
        print("Starting epoch", epoch)
        train_loss, train_accuracy = train_single_epoch(
            epoch, net, train_loader, optimizer, device, loss_function=args.loss_function, loss_mean=args.loss_mean,
        )
        test_loss, test_accuracy = test_single_epoch(
            epoch, net, val_loader, device, loss_function=args.loss_function)

        training_set_loss[epoch] = train_loss
        training_set_accuracy[epoch] = train_accuracy
        writer.add_scalar(save_name + "_train_loss", train_loss, (epoch + 1))
        writer.add_scalar(save_name + "_train_accuracy", train_accuracy, (epoch + 1))
        test_set_loss[epoch] = test_loss
        test_set_accuracy[epoch] = test_accuracy
        writer.add_scalar(save_name + "_test_loss", test_loss, (epoch + 1))
        writer.add_scalar(save_name + "_test_accuracy", test_accuracy, (epoch + 1))

        scheduler.step()

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            saved_name = args.save_loc + save_name + "_best.pth"
            torch.save(net.state_dict(), saved_name)
            print("Best model saved at epoch:", best_epoch)

        if (epoch + 1) % args.save_interval == 0:
            saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + ".pth"
            torch.save(net.state_dict(), saved_name)

    end_time = time.time()
    duration = end_time - start_time
    saved_name = args.save_loc + args.dataset + "_" + save_name + "_" + str(epoch + 1) + "_" + curr_time + ".pth"
    torch.save(net.state_dict(), saved_name)
    print("Model saved to ", saved_name)
    print("Duration: ", duration, "seconds")
    args_dict = vars(args)
    writer.close()
    with open(saved_name[: saved_name.rfind("_")] + "_train_log.json", "a") as f:
        json.dump(args_dict, f)
        json.dump(training_set_loss, f)
        json.dump(training_set_accuracy, f)
        json.dump(test_set_loss, f)
        json.dump(test_set_accuracy, f)
