
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def imagenet_dataloader(batch_size, augment, val_seed, val_size=0.1, num_workers=4, pin_memory=False, **kwargs):
    error_msg = "[!] val_size should be in the range [0, 1]."
    assert (val_size >= 0) and (val_size <= 1), error_msg

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)
    # Normalize 바꿔복디
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    # define transforms
    # to 128X128 as in CNN study
    val_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(), normalize,])

    if augment:
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,]
        )
    else:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(), normalize,])

    # load the dataset
    traindata_dir = "/home/datasets/ILSVRC12/train"
    valdata_dir = "/home/datasets/ILSVRC12/val"
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