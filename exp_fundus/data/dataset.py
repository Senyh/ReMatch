import sys
import os
import random
from skimage import io
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from exp_fundus.data import transforms as T
from torchvision.transforms import *
import numpy as np
from copy import deepcopy


class FundusDataset(Dataset):
    def __init__(self, image_path='', image_size=256, stage='train', is_augmentation=False, labeled=False, percentage=0.1, modality=''):
        super(FundusDataset, self).__init__()
        self.image_size = image_size
        self.sep = '\\' if sys.platform[:3] == 'win' else '/'
        self.stage = stage
        self.is_augmentation = is_augmentation
        self.image_path = image_path
        self.labeled = labeled
        self.modality = modality
        if self.stage == 'train':
            with open(os.path.join(self.image_path, self.modality + "_train.list"), "r") as f1:
                patient_list = f1.readlines()
            patient_list = [item.replace("\n", "") for item in patient_list]
            if labeled:
                self.sample_list = patient_list[:int(len(patient_list)*percentage)]
            else:
                self.sample_list = patient_list[int(len(patient_list)*percentage):]
        elif self.stage == 'val':
            with open(os.path.join(self.image_path, self.modality + "_test.list"), "r") as f1:
                patient_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in patient_list]
        else:
            with open(os.path.join(self.image_path, self.modality + "_train.list"), "r") as f1:
                patient_list_1 = f1.readlines()
            with open(os.path.join(self.image_path, self.modality + "_test.list"), "r") as f2:
                patient_list_2 = f2.readlines()    
            patient_list = patient_list_1 + patient_list_2
            self.sample_list = [item.replace("\n", "") for item in patient_list]
        if self.is_augmentation:
            self.augmentation = self.augmentation_transform()
        self.post_transform = self.post_transform()
        self.label_transform = self.label_transform()
        self.pre_transform = self.pre_transform()
        

    def __getitem__(self, item):
        if self.stage == 'train':
            image = io.imread(os.path.join(self.image_path, self.sample_list[item].split(',')[0])).astype('uint8')
            label_ = io.imread(os.path.join(self.image_path, self.sample_list[item].split(',')[1]).replace('.tif', '-1.tif') if self.modality == 'BinRushed' or 'Magrabia'
                              else os.path.join(self.image_path, self.sample_list[item].split(',')[1])).astype('uint8')
            # from 255(background) 128(OD) 0(OC) 
            # to 0(background) 1(OD) 2(OC) 
            label = np.zeros_like(label_)
            label[label_ < 255] = 1
            label[label_ == 0] = 2
            image = Image.fromarray(image)
            label = Image.fromarray(label).convert('L')
            image, label = self.pre_transform(image, label)
            imageA1, imageA2 = deepcopy(image), deepcopy(image)
            imageA1, _ = self.augmentation(imageA1, label)
            imageA2, _ = self.augmentation(imageA1, label)
            image, label = self.post_transform(image), self.label_transform(label)
            imageA1 = self.post_transform(imageA1)
            imageA2 = self.post_transform(imageA2)
            label = torch.from_numpy(np.array(label)).unsqueeze(0).float()
            return image, label, imageA1, imageA2
        else:
            image = io.imread(os.path.join(self.image_path, self.sample_list[item].split(',')[0])).astype('uint8')
            label_ = io.imread(os.path.join(self.image_path, self.sample_list[item].split(',')[1]).replace('.tif', '-1.tif') if self.modality == 'BinRushed' or 'Magrabia'
                              else os.path.join(self.image_path, self.sample_list[item].split(',')[1])).astype('uint8')
            # from 255(background) 128(OD) 0(OC) 
            # to 0(background) 1(OD) 2(OC) 
            label = np.zeros_like(label_)
            label[label_ < 255] = 1
            label[label_ == 0] = 2
            image = Image.fromarray(image)
            label = Image.fromarray(label).convert('L')
            image = self.post_transform(image)
            label = self.label_transform(label) if self.stage == 'val' else label
            label = torch.from_numpy(np.array(label)).unsqueeze(0).float()
            return image, label

    def __len__(self):
        return len(self.sample_list)

    @staticmethod
    def augmentation_transform():
        return T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05, p=.8), 
            T.RandomAutocontrast(p=.5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=.5),
        ])
    
    def pre_transform(self):
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=180),
        ])

    def post_transform(self):
        return Compose([
            Resize([self.image_size, self.image_size], InterpolationMode.BILINEAR),
            ToTensor(),
        ])

    def label_transform(self):
        return Compose([
            Resize([self.image_size, self.image_size], InterpolationMode.NEAREST)
        ])
    

