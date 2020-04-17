import os
import glob
import numpy as np
from scipy import misc

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
        
class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

class dataset_8s(Dataset):
    def __init__(self, root_dir, dataset_type, img_size, color_invert =True, transform=None,shuffle=False):
        self.root_dir = root_dir
        self.shuffle = shuffle
        self.color_invert = color_invert
        print(self.root_dir)
        self.transform = transform
        self.file_names = [f for f in glob.glob(os.path.join(root_dir, "*.npz")) \
                            if dataset_type in f]
        print('number of files loaded:',len(self.file_names))
        self.img_size = img_size

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"].reshape(8,16, 80, 80)
        target = data["target"]
        meta_target = data["meta_target"]
        if self.shuffle:
            for i in range(8):
                context = image[i,:8, :, :]
                choices = image[i,8:, :, :]
                indices = list(range(8))
                np.random.shuffle(indices)
                new_target = indices.index(target[i])
                new_choices = choices[indices, :, :]
                image[i] = np.concatenate((context, new_choices))
                target[i] = new_target

        if meta_target.dtype == np.int8:
            meta_target = meta_target.astype(np.uint8)
    
        del data
        if self.transform:
            image = self.transform(image)
            target = torch.tensor(target, dtype=torch.long)
            meta_target = self.transform(meta_target)

        return image, target, meta_target

class dataset(Dataset):
    def __init__(self, root_dir, dataset_type, img_size, color_invert =True, transform=None,shuffle=False):
        self.root_dir = root_dir
        self.color_invert = color_invert
        self.shuffle = shuffle
        print(self.root_dir)
        self.transform = transform
        self.file_names = [f for f in glob.glob(os.path.join(root_dir, "*.npz")) \
                            if dataset_type in f]
        print('number of files loaded:',len(self.file_names))
        self.img_size = img_size


    def __len__(self):
        return len(self.file_names)

    def get_num_data(self):
        return self.__len__()

    def __getitem__(self, idx):

        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"].reshape(16, 160, 160)


        resize_image = []
        for idx in range(0, 16):
            resize_image.append(misc.imresize(image[idx,:,:], (self.img_size, self.img_size)))
        resize_image = np.stack(resize_image)
        resize_image = resize_image/255.0

        if self.color_invert:
            resize_image = 1-resize_image

        target = data["target"]
        meta_target = data["meta_target"]


        if self.shuffle:
            context = resize_image[:8, :, :]
            choices = resize_image[8:, :, :]
            indices = list(range(8))
            np.random.shuffle(indices)
            new_target = indices.index(target)
            new_choices = choices[indices, :, :]
            resize_image = np.concatenate((context, new_choices),axis=0)
            target = new_target



        if meta_target.dtype == np.int8:
            meta_target = meta_target.astype(np.uint8)

    
        del data
        if self.transform:
            resize_image = self.transform(resize_image)
            target = torch.tensor(target, dtype=torch.long)
            meta_target = self.transform(meta_target)
        return resize_image, target, meta_target

class dataset_raven(Dataset):
    def load_subfolder_files(self,root_dir,dataset_type):
        all_files = []
        for r, d, f in os.walk(root_dir):
            for filename in f:
                if dataset_type in filename and 'npz' in filename:
                    all_files.append(os.path.join(r,filename))

        return all_files

    def __init__(self, root_dir, dataset_type, img_size, color_invert =False, transform=None, subfolder = True,shuffle=False):
        self.root_dir = root_dir
        self.color_invert = color_invert
        self.shuffle = shuffle
        print(self.root_dir)
        #print(os.path.join(root_dir,'*','*.npz'))
        self.transform = transform
        if not subfolder:
            self.file_names = [f for f in glob.glob(os.path.join(root_dir, "*.npz")) \
                            if dataset_type in f]
        else:
            self.file_names = self.load_subfolder_files(root_dir,dataset_type)

        print('number of files loaded:',len(self.file_names))
        self.img_size = img_size

    def __len__(self):
        return len(self.file_names)

    def get_num_data(self):
        return self.__len__()

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"].reshape(16, 160, 160)
        target = data["target"]
        meta_target = data["meta_target"]
        if random.randint(0,1) == 1:
            context = image[:8, :, :].copy()
            image[:3,:,:] = context[3:6,:,:]
            image[3:6,:,:] = context[:3,:,:]

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = list(range(8))
            np.random.shuffle(indices)
            new_target = indices.index(target)
            new_choices = choices[indices, :, :]
            image = np.concatenate((context, new_choices))
            target = new_target

        resize_image = []
        for idx in range(0, 16):
            resize_image.append(misc.imresize(image[idx,:,:], (self.img_size, self.img_size)))
        resize_image = np.stack(resize_image)
        resize_image = resize_image / 255.0
        if self.color_invert:
            resize_image = 1-resize_image

        if meta_target.dtype == np.int8:
            meta_target = meta_target.astype(np.uint8)
    
        del data
        if self.transform:
            resize_image = self.transform(resize_image)
            target = torch.tensor(target, dtype=torch.long)
            meta_target = self.transform(meta_target)
        return resize_image, target, meta_target
