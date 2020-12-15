import os
from PIL import Image
import torch
import random
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F


class ImageData(data.Dataset):
    def __init__(self, rgb_root, dep_root, transform):

        self.rgb_path = list(
            map(lambda x: os.path.join(rgb_root, x), os.listdir(rgb_root)))
        self.dep_path = list(
            map(
                lambda x: os.path.join(dep_root,
                                       x.split('/')[-1][:-3] + 'png'),
                self.rgb_path))

        self.transform = transform

    def __getitem__(self, item):

        rgb = Image.open(self.rgb_path[item]).convert('RGB')
        dep = Image.open(self.dep_path[item]).convert('RGB')
        [h, w] = dep.size
        imsize = [w, h]

        [rgb, dep] = self.transform(rgb, dep)

        return rgb, dep, self.rgb_path[item].split('/')[-1], imsize

    def __len__(self):
        return len(self.rgb_path)


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, rgb, dep):

        assert rgb.size == dep.size

        rgb = rgb.resize(self.size, Image.BILINEAR)
        dep = dep.resize(self.size, Image.BILINEAR)

        return rgb, dep


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, rgb, dep):

        return F.to_tensor(rgb), F.to_tensor(dep)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, rgb, dep):

        dep = F.normalize(dep, self.mean, self.std)

        rgb = F.normalize(rgb, self.mean, self.std)

        return rgb, dep


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, rgb, dep):
        if random.random() < self.p:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            dep = dep.transpose(Image.FLIP_LEFT_RIGHT)

        return rgb, dep


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb, dep):
        for t in self.transforms:
            rgb, dep = t(rgb, dep)
        return rgb, dep

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def get_loader(rgb_root,
               dep_root,
               img_size,
               batch_size=1,
               num_thread=1,
               pin=False):
    test_transform = Compose([
        FixedResize(img_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageData(rgb_root, dep_root, transform=test_transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_thread,
                                  pin_memory=pin)
    return data_loader
