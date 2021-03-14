#from __future__ import print_function, division

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def make_dataset(images_file_path, labels):
    image_list = open(images_file_path).readlines()
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class TmageList(Dataset):
    def __init__(
        self,
        images_file_path,
        transform=None,
        target_transform=None,
        labels=None,
        loader=pil_loader,
    ):
        imgs = make_dataset(images_file_path, labels)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in the dataset")

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def _load_one_item(self, idx: int):
        path, target = self.imgs[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __getitem__(self, index):
        if isinstance(index, int):
            sample, target = self._load_one_item(index)
        elif isinstance(index, list):
            assert len(index) > 0
            samples = [None for _ in range(len(index))]
            targets = [None for _ in range(len(index))]
            for i, idx in enumerate(index):
                sample, target = self._load_one_item(idx)
                samples[i] = sample
                targets[i] = target
            sample = torch.stack(samples)
            target = torch.LongTensor(targets)
        else:
            raise ValueError('Input index should be of type int or list(int)!')

        return sample, target

    def __len__(self):
        return len(self.imgs)
