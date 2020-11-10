"""
Customized data transforms
"""
import random

from PIL import Image
from scipy import ndimage
import numpy as np
import torch
import torchvision.transforms.functional as tr_F

import math
import numbers
import collections
import cv2


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
        img, img_noresize, label, label_noresize = sample['image'], sample['image_noresize'], sample['label'], sample['label_noresize']
        img = tr_F.resize(img, self.size)
        img_noresize = tr_F.resize(img_noresize, self.size)
        if isinstance(label, dict):
            label = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
                     for catId, x in label.items()}
            label_noresize = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
                     for catId, x in label_noresize.items()}
        else:
            label = tr_F.resize(label, self.size, interpolation=Image.NEAREST)
            label_noresize = tr_F.resize(label_noresize, self.size, interpolation=Image.NEAREST)

        sample['image'] = img
        sample['image_noresize'] = img_noresize
        sample['label'] = label
        sample['label_noresize'] = label_noresize
        return sample

class Resize_test(object):
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

class ToTensorNormalize_noresize(object):
    """
    Convert images/masks to torch.Tensor
    Scale images' pixel values to [0-1] and normalize with predefined statistics
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = tr_F.to_tensor(img)
        img = tr_F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if isinstance(label, dict):
            label = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in label.items()}
        else:
            label = torch.Tensor(np.array(label)).long()

        sample['image'] = img
        sample['image_noresize'] = img
        sample['label'] = label
        sample['label_noresize'] = label

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


class Resize_pad(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        def find_new_hw(ori_h, ori_w, test_size):
            if ori_h >= ori_w:
                ratio = test_size*1.0 / ori_h
                new_h = test_size
                new_w = int(ori_w * ratio)
            elif ori_w > ori_h:
                ratio = test_size*1.0 / ori_w
                new_h = int(ori_h * ratio)
                new_w = test_size

            if new_h % 8 != 0:
                new_h = (int(new_h /8))*8
            else:
                new_h = new_h
            if new_w % 8 != 0:
                new_w = (int(new_w /8))*8
            else:
                new_w = new_w    
            return new_h, new_w           

        test_size = self.size
        new_h, new_w = find_new_hw(image.shape[0], image.shape[1], test_size)
        #new_h, new_w = test_size, test_size
        image_crop = cv2.resize(image, dsize=(int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
        back_crop = np.zeros((test_size, test_size, 3)) 
        # back_crop[:,:,0] = mean[0]
        # back_crop[:,:,1] = mean[1]
        # back_crop[:,:,2] = mean[2]
        back_crop[:new_h, :new_w, :] = image_crop
        image = back_crop 

        s_mask = label
        new_h, new_w = find_new_hw(s_mask.shape[0], s_mask.shape[1], test_size)
        #new_h, new_w = test_size, test_size
        s_mask = cv2.resize(s_mask.astype(np.float32), dsize=(int(new_w), int(new_h)),interpolation=cv2.INTER_NEAREST)
        back_crop_s_mask = np.ones((test_size, test_size)) * 255
        back_crop_s_mask[:new_h, :new_w] = s_mask
        label = back_crop_s_mask

        sample['image'], sample['label'] = image, label
        return sample

class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        sample['image'], sample['label'] = image, label
        return sample

class ToNumpy(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.asarray(image)
        label = np.asarray(label)
        sample['image'] = image
        sample['label'] = label
        del sample['image_noresize']
        del sample['label_noresize']
        return sample


class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        self.size = size
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = label.shape

        
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape
        raw_label = label
        raw_image = image

        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        raw_pos_num = np.sum(raw_label == 1)
        pos_num = np.sum(label == 1)
        crop_cnt = 0
        while(pos_num < 0.85*raw_pos_num and crop_cnt<=30):
            image = raw_image
            label = raw_label
            if self.crop_type == 'rand':
                h_off = random.randint(0, h - self.crop_h)
                w_off = random.randint(0, w - self.crop_w)
            else:
                h_off = int((h - self.crop_h) / 2)
                w_off = int((w - self.crop_w) / 2)
            image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
            label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]   
            raw_pos_num = np.sum(raw_label == 1)
            pos_num = np.sum(label == 1)  
            crop_cnt += 1
        if crop_cnt >= 50:
            image = cv2.resize(raw_image, (self.size[0], self.size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(raw_label, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)            
                               
        if image.shape != (self.size[0], self.size[0], 3):
            image = cv2.resize(image, (self.size[0], self.size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)

        sample['image'], sample['label'] = image, label
        return sample


class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        sample['image'], sample['label'] = image, label
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        sample['image'], sample['label'] = image, label
        return sample


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        sample['image'], sample['label'] = image, label
        return sample


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        sample['image'], sample['label'] = image, label
        return sample
