import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import datasets, utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

path2data = "c:/Users/FS/Desktop/JHW/datasets/classification"

if not os.path.exists(path2data):
    os.mkdir(path2data)
train_ds = datasets.STL10(path2data, split='train', download=True, transform=transforms.ToTensor())
val_ds = datasets.STL10(path2data, split='test', download=True, transform=transforms.ToTensor())

def data_transform(train_ds, val_ds):
    train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_ds]
    train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_ds]

    train_meanR = np.mean([m[0] for m in train_meanRGB])
    train_meanG = np.mean([m[1] for m in train_meanRGB])
    train_meanB = np.mean([m[2] for m in train_meanRGB])
    train_stdR = np.mean([s[0] for s in train_stdRGB])
    train_stdG = np.mean([s[1] for s in train_stdRGB])
    train_stdB = np.mean([s[2] for s in train_stdRGB])

    val_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in val_ds]
    val_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in val_ds]

    val_meanR = np.mean([m[0] for m in val_meanRGB])
    val_meanG = np.mean([m[1] for m in val_meanRGB])
    val_meanB = np.mean([m[2] for m in val_meanRGB])

    val_stdR = np.mean([s[0] for s in val_stdRGB])
    val_stdG = np.mean([s[1] for s in val_stdRGB])
    val_stdB = np.mean([s[2] for s in val_stdRGB])
    
    train_transformation = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize(224),
                        transforms.Normalize([train_meanR, train_meanG, train_meanB],[train_stdR, train_stdG, train_stdB]),
                        transforms.RandomHorizontalFlip(),
    ])

    val_transformation = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(224),
                            transforms.Normalize([val_meanR, val_meanG, val_meanB],[val_stdR, val_stdG, val_stdB]),
    ])

    train_ds.transform = train_transformation
    val_ds.transform = val_transformation

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=True)
    
    return train_dl, val_dl