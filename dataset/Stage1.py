
import torch
from torch.utils.data.dataset import Dataset
import os
import random
import glob
import torchio as tio
import json
import random

class VQGANDataset(Dataset):
    def __init__(self, root_dir=None, augmentation=False,split='train',patch_size = 128):
        randnum = 216
        self.file_names = []

        print(root_dir)
        if root_dir.endswith('json'):
            with open(root_dir) as json_file:
                dataroots = json.load(json_file)
            for key,value in dataroots.items():
                if type(value) == list:
                    for path in value:
                        self.file_names += (glob.glob(os.path.join(path, './*.nii.gz'), recursive=True))
                else:
                    self.file_names += (glob.glob(os.path.join(value, './*.nii.gz'), recursive=True))
        else:
            self.root_dir = root_dir
            self.file_names = glob.glob(os.path.join(
                        root_dir, './*.nii.gz'), recursive=True)
        random.seed(randnum)
        random.shuffle(self.file_names )

        self.split = split
        self.augmentation = augmentation
        if split == 'train':
            self.file_names = self.file_names[:-10]
            print(self.file_names[:100])
            print(len(self.file_names))
        elif split == 'val':
            self.file_names = self.file_names[-10:]
            self.augmentation = False #Double check
            print(len(self.file_names))
        self.patch_sampler = tio.data.UniformSampler(patch_size)

        self.patch_sampler_256 = tio.data.UniformSampler((256,256,128))#
        self.randomflip = tio.RandomFlip( axes=(0,1),flip_probability=0.5)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]
        whole_img = tio.ScalarImage(path)
        if  self.split == 'train':
            img = None
            while img== None or img.data.sum() ==0:
                img = next(self.patch_sampler(tio.Subject(image = whole_img)))['image']

        elif self.split =='val':
            img = whole_img
            if img.shape[1]*img.shape[2]*img.shape[3] > 256*256*128:#
                img = next(self.patch_sampler_256(tio.Subject(image = img)))['image']
        if self.augmentation:
            img = self.randomflip(img)
        imageout = img.data 
        if self.augmentation and random.random()>0.5:
            imageout = torch.rot90(imageout,dims=(1,2))
            
        imageout = imageout * 2 - 1
        imageout = imageout.transpose(1,3).transpose(2,3)
        imageout = imageout.type(torch.float32)

        if self.split =='val':
            return {'data': imageout , 'affine' : img.affine , 'path':path}
        else:
            return {'data': imageout}
        # return img,path