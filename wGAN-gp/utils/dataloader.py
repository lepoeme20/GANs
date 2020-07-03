import os
import torch
import torchvision
from torchvision import transforms


def get_dataloader(args):
    transformer = __get_transformer(args)
    dataset = __get_dataset_name(args)
    trn_loader, dev_loader, tst_loader = __get_loader(args, dataset, transformer)

    return trn_loader, dev_loader, tst_loader

def __get_loader(args, data_name, transformer):
    root = args.data_root_path
    data_path = os.path.join(root, args.dataset.lower())
    dataset = getattr(torchvision.datasets, data_name)

    trn_transform, tst_transform = transformer
    # call dataset
    if data_name == 'CelebA':
        trainset = dataset(root=data_path, download=True, split='train', transform=trn_transform)
        devset = dataset(root=data_path, download=True, split='valid', transform=tst_transform)
        tstset = dataset(root=data_path, download=True, split='test', transform=tst_transform)
    else:
        trainset = dataset(root=data_path, download=True, train=True, transform=trn_transform)
        _devset = dataset(data_path, download=True, train=False, transform=tst_transform)
        devset, tstset = torch.utils.data.random_split(
            _devset, [int(len(_devset) * .2), int(len(_devset) * .8)])

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    devloader = torch.utils.data.DataLoader(
        devset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)
    tstloader = torch.utils.data.DataLoader(
        tstset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)

    return trainloader, devloader, tstloader


def __get_transformer(args):
    if args.dataset.lower() == 'mnist':
        m, s = [0.1307], [0.3081]
    elif args.dataset.lower() == 'cifar10':
        m, s = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    elif args.dataset.lower() == 'fmnist':
        m, s = [0.5], [0.5]
    elif args.dataset.lower() == 'celeba':
        m, s = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    trn_transformer = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=m, std=s
        ), ])

    dev_transformer = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=m, std=s
        ), ])

    return trn_transformer, dev_transformer

def __get_dataset_name(args):
    if args.dataset.lower() == 'mnist':
        d_name = 'MNIST'
    elif args.dataset.lower() == 'fmnist':
        d_name = 'FashionMNIST'
    elif args.dataset.lower() == 'celeba':
        d_name = 'CelebA'
    elif args.dataset.lower() == 'cifar10':
        d_name = 'CIFAR10'

    return d_name