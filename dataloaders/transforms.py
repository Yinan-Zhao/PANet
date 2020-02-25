"""
Customized data transforms
"""
import random

from PIL import Image
from scipy import ndimage
import numpy as np
import torch
import torchvision.transforms.functional as tr_F


class RandomMirror(object):
    """
    Randomly filp the images/masks horizontally
    """
    def __call__(self, sample):
        img, img_noresize, label, label_noresize = sample['image'], sample['image_noresize'], sample['label'], sample['label_noresize']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_noresize = img_noresize.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(label, dict):
                label = {catId: x.transpose(Image.FLIP_LEFT_RIGHT)
                         for catId, x in label.items()}
                label_noresize = {catId: x.transpose(Image.FLIP_LEFT_RIGHT)
                         for catId, x in label_noresize.items()}
            else:
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
                label_noresize = label_noresize.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = img
        sample['image_noresize'] = img_noresize
        sample['label'] = label
        sample['label_noresize'] = label_noresize
        return sample

class Resize(object):
    """
    Resize images/masks to given size

    Args:
        size: output size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = tr_F.resize(img, self.size)
        if isinstance(label, dict):
            label = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
                     for catId, x in label.items()}
        else:
            label = tr_F.resize(label, self.size, interpolation=Image.NEAREST)

        sample['image'] = img
        sample['label'] = label
        return sample

class ToTensorNormalize(object):
    """
    Convert images/masks to torch.Tensor
    Scale images' pixel values to [0-1] and normalize with predefined statistics
    """
    def __call__(self, sample):
        img, img_noresize, label, label_noresize = sample['image'], sample['image_noresize'], sample['label'], sample['label_noresize']
        img = tr_F.to_tensor(img)
        img = tr_F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_noresize = tr_F.to_tensor(img_noresize)
        img_noresize = tr_F.normalize(img_noresize, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if isinstance(label, dict):
            label = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in label.items()}
            label_noresize = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in label_noresize.items()}
        else:
            label = torch.Tensor(np.array(label)).long()
            label_noresize = torch.Tensor(np.array(label_noresize)).long()

        sample['image'] = img
        sample['image_noresize'] = img_noresize
        sample['label'] = label
        sample['label_noresize'] = label_noresize

        return sample
