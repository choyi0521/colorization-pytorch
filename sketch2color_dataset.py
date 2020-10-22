from torch.utils.data import Dataset
import os
from os import listdir
from PIL import Image, ImageCms
from torchvision import transforms
import cv2
import numpy as np
from albumentations import (
    HorizontalFlip, RandomResizedCrop, Resize, OneOf, Compose, RandomBrightnessContrast, HueSaturationValue
)
from albumentations.pytorch.transforms import ToTensor


def get_train_transform(args):
    return Compose([
        OneOf([
            RandomResizedCrop(args.input_height, args.input_width, interpolation=cv2.INTER_NEAREST),
            RandomResizedCrop(args.input_height, args.input_width, interpolation=cv2.INTER_LINEAR),
            RandomResizedCrop(args.input_height, args.input_width, interpolation=cv2.INTER_CUBIC),
            RandomResizedCrop(args.input_height, args.input_width, interpolation=cv2.INTER_AREA),
            RandomResizedCrop(args.input_height, args.input_width, interpolation=cv2.INTER_LANCZOS4),
            Resize(args.input_height, args.input_width)
        ], p=1.0),
        HorizontalFlip(),
        ToTensor()
    ])


def get_test_transform(args):
    return Compose([
        Resize(args.input_height, args.input_width),
        ToTensor()
    ])

class Sketch2ColorDataset(Dataset):
    def __init__(self, dataset_dir, phase):
        super().__init__()
        self.phase = phase
        if phase not in ['train', 'val', 'test']:
            raise Exception('No such dataset_type')
        
        if phase in ['train', 'val']:
            image_dir = os.path.join(dataset_dir, 'train')
        else:
            image_dir = os.path.join(dataset_dir, 'test')

        fnames = sorted(listdir(image_dir), key=lambda x:int(x.split('_')[0]))
        fnames_a = [image_dir+'/'+fname for fname in fnames if fname.split('_')[1][0] == 'A']
        fnames_b = [image_dir+'/'+fname for fname in fnames if fname.split('_')[1][0] == 'B']
        val_images = int(len(fnames_a)*args.val_ratio)

        if phase == 'train':
            self.fnames_a = fnames_a[:-val_images]
            self.fnames_b = fnames_b[:-val_images]
            transform = get_train_transform(args)
        elif phase == 'val':
            self.fnames_a = fnames_a[-val_images:]
            self.fnames_b = fnames_b[-val_images:]
            transform = get_test_transform(args)
        else:
            self.fnames_a = fnames_a
            self.fnames_b = fnames_b
            transform = get_test_transform(args)
        self.transform = Compose(transform, additional_targets={'image2': 'image'})

    def __getitem__(self, idx):
        img_a = cv2.imread(self.fnames_a[idx])
        img_b = cv2.imread(self.fnames_b[idx])
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        if self.phase == 'train':
            transform1, transform2 = self.transform
            img_b = transform1(image=img_b)['image']
            aug_output = transform2(image = img_a, image2 = img_b)
            return aug_output['image'], aug_output['image2']
        
        aug_output = self.transform(image = img_a, image2 = img_b)
        return aug_output['image'], aug_output['image2']

    def __len__(self):
        return len(self.fnames_a)
