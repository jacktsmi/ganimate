import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import models as md
import dataset as ds

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

batch_size = 1
num_epochs = 150
save_dir = "save"
input_size = 128

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainA = ds.ImageData("dataA", split="train", transform=transform)
testA = ds.ImageData("dataA", split="test", transform=transform)
trainB = ds.ImageData("dataB", split="train", transform=transform)
testB = ds.ImageData("dataB", split="test", transform=transform)

trainA_loader = DataLoader(trainA, batch_size=batch_size, shuffle=True)
testA_loader = DataLoader(testA, batch_size=batch_size, shuffle=False)
trainA_loader = DataLoader(trainB, batch_size=batch_size, shuffle=True)
testA_loader = DataLoader(testB, batch_size=batch_size, shuffle=False)

print('Total images in real face train set:' + str(len(trainA)))
print('Total images in real face test set:' + str(len(testA)))

print('Total images in cartoon face train set:' + str(len(trainB)))
print('Total images in cartoon face test set:' + str(len(testB)))

G_A2B = md.Generator().cuda()
G_B2A = md.Generator().cuda()
D_A = md.Discriminator().cuda()
D_B = md.Discriminator().cuda()

G_A2B.apply(weight_init_g)
G_B2A.apply(weight_init_g)
D_A.apply(weight_init_d)
D_B.apply(weight_init_d)