import random
from torch.utils.data import ConcatDataset
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from continual_datasets.continual_datasets import *
import utils
import numpy as np


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())

    return transforms.Compose(t)


def get_dataset(dataset, transform_train, transform_val, args):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'CUB200':
        dataset_train = CUB200(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'MiniImagenet':
        dataset_train = datasets.ImageFolder("./local_datasets/miniimagenet/train", transform=transform_train)
        dataset_val = datasets.ImageFolder("./local_datasets/miniimagenet/val", transform=transform_val)
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))

    return dataset_train, dataset_val


def split_dataset(dataset_train, dataset_val, args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.n_classes == 200:
        new_n_classes = 100
    elif args.n_classes == 100:
        new_n_classes = 40
    assert new_n_classes // (args.n_tasks - 1) == args.ways
    ways = args.ways
    base_labels = [i for i in range(len(dataset_train.classes) - new_n_classes)]
    labels = [i for i in range(len(dataset_train.classes) - new_n_classes, len(dataset_train.classes))]

    split_datasets = list()
    mask = list()

    random.shuffle(labels)

    def select_random_elements(lst, n):
        result = []
        unique_elements = set(lst)  # 获取列表中的唯一元素
        for element in unique_elements:
            indices = [i for i, x in enumerate(lst) if x == element]  # 找到元素在列表中的所有位置
            selected_indices = random.sample(indices, n)  # 从位置中随机选择n个位置
            result.extend(selected_indices)  # 将选中的位置添加到结果列表中
        return result

    def get_corresponding_values(lst, positions):
        result = [lst[i] for i in positions]
        return result

    for t in range(args.n_tasks):
        train_split_indices = []
        test_split_indices = []
        pos_mask = []
        if t != 0:
            scope = labels[:ways]
            labels = labels[ways:]
            mask.append(scope)
        else:
            mask.append(base_labels)
        if t == 0:
            for k in range(len(dataset_train.targets)):
                if int(dataset_train.targets[k]) in base_labels:
                    train_split_indices.append(k)
            for h in range(len(dataset_val.targets)):
                if int(dataset_val.targets[h]) in base_labels:
                    test_split_indices.append(h)
        else:
            for k in range(len(dataset_train.targets)):
                if int(dataset_train.targets[k]) in scope:
                    train_split_indices.append(k)
                    pos_mask.append(int(dataset_train.targets[k]))
                    # count_list[dataset_train.targets[k]]+=1
            for h in range(len(dataset_val.targets)):
                if int(dataset_val.targets[h]) in scope:
                    test_split_indices.append(h)
        if t == 0:
            subset_train, subset_val = Subset(dataset_train, train_split_indices), Subset(dataset_val,
                                                                                          test_split_indices)
        else:
            pos = select_random_elements(pos_mask, args.shots)
            few_train_split_indices = get_corresponding_values(train_split_indices, pos)
            subset_train, subset_val = Subset(dataset_train, few_train_split_indices), Subset(dataset_val,
                                                                                              test_split_indices)

        split_datasets.append([subset_train, subset_val])

    return split_datasets, mask


def dataloader(args):
    dataloader = list()

    # set transform
    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    # Load local_datasets
    dataset_train, dataset_val = get_dataset(args.dataset, transform_train, transform_val, args)

    splited_dataset, class_mask = split_dataset(dataset_train, dataset_val, args)

    for i in range(args.n_tasks):
        dataset_train, dataset_val = splited_dataset[i]
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )
        dataloader.append({'train': data_loader_train, 'val': data_loader_val})

    return dataloader, class_mask
