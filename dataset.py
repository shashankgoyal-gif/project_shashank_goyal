# dataset.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import *

class MushroomDataset(datasets.ImageFolder):
    def __init__(self, root_dir, phase='train'):
        if phase == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(CROP_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(RESIZE_X),
                transforms.CenterCrop(CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ])
        # Calling the parent class to handle the directory reading
        super().__init__(os.path.join(root_dir, phase), transform=transform)

def get_mushroom_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)