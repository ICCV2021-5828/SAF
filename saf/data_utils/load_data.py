
# import os
# import sys

# from random import shuffle
import torch
from torch import nn
# import torchvision as tv
# from torchvision.datasets import ImageFolder
import torchvision.transforms as T
# from tqdm import tqdm

from .. import cfg
from .sampler import get_sampler
from .TensorFolder import TensorFolder
from .TensorLoader import TensorLoader
from .TmageFolder import TmageFolder
from .TmageList import TmageList


def getTransforms(is_train, raw_image=False, args=None):
    if args is None:
        args = cfg.args
    Normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms_list = []
    if args.resize_size != 256 or raw_image:
        transforms_list.append(T.Resize(
            (args.resize_size, args.resize_size)
        ))
    if is_train:
        if args.is_cen or args.crop_type == 'CenterCrop':
            transforms_list.append(T.CenterCrop(args.crop_size))
        elif args.crop_type == 'RandomResizedCrop':
            transforms_list.append(T.RandomResizedCrop(args.crop_size))
        elif args.crop_type == 'RandomCrop':
            transforms_list.append(T.RandomCrop(args.crop_size))
        else:
            raise NotImplementedError
        transforms_list.append(T.RandomHorizontalFlip())
    else:
        transforms_list.append(T.CenterCrop(args.crop_size))

    if raw_image:
        transforms_list.append(T.ToTensor())
        transforms_list.append(Normalizer)
        transforms = T.Compose(transforms_list)
    else:
        transforms_list.append(Normalizer)
        transforms = nn.Sequential(*transforms_list)

    return transforms


def loadImageTensor(
    root, batch_size, is_source=True,
    tensor_folder=False, image_folder=False, image_list=False,
    is_train=True, drop_last=False, num_workers=0,
    args=None,
):
    transforms = getTransforms(is_train, raw_image=(image_folder or image_list), args=args)

    if tensor_folder:
        tensor_dataset = TensorFolder(root, transform=transforms)
    elif image_folder:
        tensor_dataset = TmageFolder(root, transform=transforms)
    elif image_list:
        tensor_dataset = TmageList(root, transform=transforms)
    else:
        tensor_dataset = TensorLoader(root, transform=transforms)

    cfg.logger.info(f'Loading data from {root} ...')
    sampler_arg = get_sampler(tensor_dataset, is_source=is_source) if is_train else None
    batch_size_arg = batch_size if (sampler_arg is None) else 1
    shuffle_arg = is_train and (sampler_arg is None)
    drop_last_arg = drop_last and is_train and (sampler_arg is None)

    image_loader = torch.utils.data.DataLoader(
        tensor_dataset,
        batch_size=batch_size_arg,
        shuffle=shuffle_arg,
        sampler=sampler_arg,
        num_workers=num_workers,
        drop_last=drop_last_arg,
    )
    return image_loader

