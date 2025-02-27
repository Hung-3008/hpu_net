#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Apply augmentations randomly.
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # Resize if necessary.
        x, y = image.shape
        target_x, target_y = self.target_size
        if (x, y) != self.target_size:
            zoom_factors = (target_x / x, target_y / y)
            image = zoom(image, zoom_factors, order=3)   # Using cubic interpolation for image
            label = zoom(label, zoom_factors, order=0)     # Nearest-neighbor for labels

        # Add channel dimension to image.
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class ACDCdataset(Dataset):
    def __init__(self, base_dir, list_dir, split, target_size=(256, 256), transform=None):
        """
        Parameters:
          base_dir: base directory where data is stored.
          list_dir: directory containing text files with sample names.
          split: 'train', 'valid', or 'test' indicating the dataset split.
          target_size: desired output size of (height, width)
          transform: transformation to be applied on a sample (e.g., RandomGenerator).
        """
        self.split = split
        self.data_dir = base_dir
        self.target_size = target_size
        
        # Read the list of sample file names.
        list_path = os.path.join(list_dir, split + '.txt')
        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]
        
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx]
        if self.split in ["train", "valid"]:
            data_path = os.path.join(self.data_dir, self.split, sample_name)
        else:
            data_path = os.path.join(self.data_dir, sample_name)
        
        data = np.load(data_path)
        image, label = data['img'], data['label']
        
        # Resize if transform is not provided.
        if self.transform is None:
            x, y = image.shape
            target_x, target_y = self.target_size
            if (x, y) != self.target_size:
                zoom_factors = (target_x / x, target_y / y)
                image = zoom(image, zoom_factors, order=3)
                label = zoom(label, zoom_factors, order=0)
            
            # Add channel dimension (for image only)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.float32)).long()
            sample = {'image': image, 'label': label}
        else:
            # Apply transform (which should include resizing & adding channel)
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)

        # Save the case name if needed.
        sample['case_name'] = sample_name
        return sample
