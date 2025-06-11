
import torch
from torch.utils.data.dataset import Dataset
import os
import random
import glob
import torchio as tio
import json
import random
from os.path import join
class DenoisingDataset(Dataset):
    def __init__(self, input_dir=None, gt_dir=None , augmentation=False,split='train'):
        self.augmentation = augmentation
        self.input_dir = input_dir
        self.split = split
        self.input_names = glob.glob(
            input_dir+ '/*.nii.gz', recursive=True)

        self.gt_dir = gt_dir 
        self.input_names.sort()
        if split == 'train':
            self.input_names = self.input_names[:-1]
        elif split == 'val':
            self.input_names = self.input_names[-1:]
        else:
            self.input_names = self.input_names
            self.augmentation = False
        self.patch_sampler = tio.data.UniformSampler(128)
        print(len(self.input_names))

    def __len__(self):

        return len(self.input_names)


    def __getitem__(self, index):

        input_path = self.input_names[index]
        if self.split =='train' or self.split == 'val':
            gt_name = os.path.basename(input_path).replace('_1_6','')
            gt_path = join(self.gt_dir,gt_name)
            input_img ,gt_img = tio.ScalarImage(input_path) , tio.ScalarImage(gt_path)

            patch = next(self.patch_sampler(tio.Subject(input = input_img, gt = gt_img)))
            if self.augmentation:
                transform = tio.Compose([
                    tio.RandomFlip(axes=(0, 1), flip_probability=0.5),
                ])
                patch = transform(patch)
            input_patch = patch['input']
            gt_patch = patch['gt']
            data_in, data_gt = input_patch.data , gt_patch.data
            data_in, data_gt = data_in * 2 - 1 , data_gt * 2 - 1
            data_in= data_in.transpose(1,3).transpose(2,3)
            data_gt = data_gt.transpose(1,3).transpose(2,3)
        else:
            input_img  = tio.ScalarImage(input_path)
            data_in = input_img.data 
            data_in = data_in * 2 - 1 
            data_in= data_in.transpose(1,3).transpose(2,3)


        if self.split =='val':
            return {'input': data_in, 'target':data_gt ,'affine' : input_img.affine , 'path':input_path}
        elif self.split =='train':
            return {'input': data_in, 'target':data_gt}
        else:
            return {'input': data_in,'affine' : input_img.affine , 'path':input_path}
