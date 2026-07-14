import sys
import os
import random
import torch
from torch.utils.data.dataset import Dataset
from exp_mscmr.data import transforms as T
from torchvision.transforms import *
import numpy as np
import h5py
from PIL import Image
from copy import deepcopy


class MSCMRDataset(Dataset):
    def __init__(self, image_path='', image_size=256, stage='train', modality='C0', is_augmentation=False, labeled=False, percentage=0.1):
        super(MSCMRDataset, self).__init__()
        self.image_size = image_size
        self.sep = '\\' if sys.platform[:3] == 'win' else '/'
        self.stage = stage
        self.is_augmentation = is_augmentation
        self.image_path = image_path
        self.labeled = labeled
        
        if self.stage == 'train':
            with open(self.image_path + "/train36.list", "r") as f1:
                patient_list = f1.readlines()
            patient_list = [item.replace("\n", "") for item in patient_list]
            if labeled:
                patient_list = patient_list[:int(len(patient_list)*percentage)]
            else:
                patient_list = patient_list[int(len(patient_list)*percentage):]
            train_slices_list = [item for item in os.listdir(os.path.join(self.image_path, 'MS_CMR_OriSize_h5py')) if item.find(modality) != -1]
            self.sample_list = [x for y in patient_list for x in train_slices_list if x.startswith(y + '_')]
        else:
            with open(self.image_path + "/val9.list", "r") as f1:
                patient_list = f1.readlines()
            patient_list = [item.replace("\n", "") for item in patient_list]
            val_slices_list = [item for item in os.listdir(os.path.join(self.image_path, 'MS_CMR_OriSize_h5py')) if item.find(modality) != -1]
            self.sample_list = [x for y in patient_list for x in val_slices_list if x.startswith(y + '_')]
        if self.is_augmentation:
            self.augmentation = self.augmentation_transform()
        self.post_transform = self.post_transform()
        self.label_transform = self.label_transform()
        self.pre_transform = self.pre_transform()

    def __getitem__(self, item):
        
        if self.stage == 'train':
            case = self.sample_list[item]
            h5f = h5py.File(os.path.join(self.image_path, 'MS_CMR_OriSize_h5py', case), "r")
            image = h5f["image"][:] * 255.
            label = h5f["label"][:]
            image = Image.fromarray(image).convert('L')
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
            case = self.sample_list[item]
            h5f = h5py.File(os.path.join(self.image_path, 'MS_CMR_OriSize_h5py', case), "r")
            image = h5f["image"][:] * 255.
            label = h5f["label"][:]
            image = Image.fromarray(image).convert('L')
            label = Image.fromarray(label).convert('L')
            image, label = self.post_transform(image), self.label_transform(label)
            label = torch.from_numpy(np.array(label)).unsqueeze(0).float()
            return image, label

    def __len__(self):
        return len(self.sample_list)

    @staticmethod
    def augmentation_transform():
        return T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0., p=.5),
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



    

