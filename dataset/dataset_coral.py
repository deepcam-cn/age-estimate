# coding: utf-8

import os
import cv2
import random
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import Dataset

class Dataset(Dataset):
    """Custom Dataset for loading face images"""

    def __init__(self, root, data_list_file, num_classes=100, phase='train', input_shape=(3, 112, 112)):
        self.phase = phase
        self.input_shape = input_shape
        self.NUM_CLASSES = num_classes

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()
        
        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize(self.input_shape[1:]),
                T.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def random_crop(self, img, min_scale = 0.85):
        rand_scale = random.randint(min_scale * 100, 100) / 100.0
        new_size = int(self.input_shape[-1] / rand_scale)
        new_img = T.Resize((new_size, new_size))(img)
        new_img = T.RandomCrop(self.input_shape[-1])(new_img)
        return new_img

    def rand_augment(self, img):
        # random flip
        img = img.resize(self.input_shape[1:])
        if self.phase == 'test':
            return img

        rand_num = random.randint(0, 1)
        if rand_num == 1:
            img = F.hflip(img)
        # random crop
        rand_num = random.randint(0, 9)
        if rand_num == 1:
            img = self.random_crop(img)
        # random gray
        rand_num = random.randint(0, 9)
        if rand_num == 1:
            img = T.Grayscale(num_output_channels=3)(img)
        return img

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]

        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.rand_augment(data)
        data = self.transforms(data)

        label = np.int32(splits[1])
        if label > (self.NUM_CLASSES - 1): 
            label = (self.NUM_CLASSES - 1)
        levels = [1]*label + [0]*(self.NUM_CLASSES - 1 - label)
        levels = torch.tensor(levels, dtype=torch.float32)

        return data, label, levels

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    dataset = Dataset(root='./',
                      data_list_file='./deepcam_coral_train_list.txt',
                      num_classes=100,
                      phase='train',
                      input_shape=(3, 112, 112))

    trainloader = data.DataLoader(dataset, batch_size=32)
    for i, (data, label, level) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        #img *= np.array([0.229, 0.224, 0.225])
        #img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        #cv2.imshow('img', img)
        #cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)
