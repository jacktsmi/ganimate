from models import Discriminator, Generator, weights_init, ImageBuffer
from dataset import ImageData

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import random

import torch
import torchvision
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchsummary import summary
import wget
import zipfile

# Download data
# dataA_url = 'https://www.dropbox.com/s/l8ltecy8qonzbcc/dataA.zip?dl=1' # Human faces
# wget.download(dataA_url, out='./dataA.zip')

# dataB_url = 'https://www.dropbox.com/s/oa6dm9a2y7ymmdx/bitmoji.zip?dl=1' # Cartoon faces
# wget.download(dataB_url, out='./dataB.zip')

# with zipfile.ZipFile('dataA.zip', 'r') as zip_ref:
#     zip_ref.extractall('dataA')

# with zipfile.ZipFile('dataB.zip', 'r') as zip_ref2:
#     zip_ref2.extractall('dataB')

# Learning Parameters
batch_size = 1 # From CyclGAN paper

num_epochs = 31

imgsize = 128

num_patches = 4

in_channels =  3

learning_rate = 0.0002

betas = (0.5, 0.999)

lam = 10

input_size = 128

# Load in Data and Initialize Networks
device = 'cpu'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only. If you want to enable GPU, please to go Edit > Notebook Settings > Hardware Accelerator and select GPU.")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = ImageData(split="train", transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = ImageData(split="test", transform=transform)

test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

print('Total images in train set:' + str(len(train_data)))
print('Total images in test set:' + str(len(test_data)))

G_A2B = Generator().cuda()
G_B2A = Generator().cuda()
D_A = Discriminator().cuda()
D_B = Discriminator().cuda()

# Pre-Saved Weights
# G_A2B.load_state_dict(torch.load('drive/MyDrive/weights/G_A2B_epoch31_new.pth'))
# G_B2A.load_state_dict(torch.load('drive/MyDrive/weights/G_B2A_epoch31_new.pth'))
# D_A.load_state_dict(torch.load('drive/MyDrive/weights/D_A_epoch31_new.pth'))
# D_B.load_state_dict(torch.load('drive/MyDrive/weights/D_B_epoch31_new.pth'))

# Normal Initialization
G_A2B.apply(weights_init)
G_B2A.apply(weights_init)
D_A.apply(weights_init)
D_B.apply(weights_init)

summary(G_A2B, (in_channels, imgsize, imgsize))
summary(G_B2A, (in_channels, imgsize, imgsize))
summary(D_A, (in_channels, imgsize, imgsize))
summary(D_B, (in_channels, imgsize, imgsize))

# Train
G_A2B_loss_traj = []
G_B2A_loss_traj = []
D_A_loss_traj = []
D_B_loss_traj = []

# Optimizers
G_optim = optim.Adam(list(G_A2B.parameters())+list(G_B2A.parameters()),
                                lr=learning_rate, betas=betas)

# Only train G_B2A for last two epochs
#G_optim = optim.Adam(list(G_B2A.parameters()), lr=learning_rate, betas=betas)
D_A_optim = optim.Adam(list(D_A.parameters()), lr=learning_rate, betas=betas)
D_B_optim = optim.Adam(list(D_B.parameters()), lr=learning_rate, betas=betas)

cycle_loss = nn.L1Loss().to(device)
adversarial_loss = nn.MSELoss().to(device)

cur_epoch = 1

fake_A_buffer = ImageBuffer()
fake_B_buffer = ImageBuffer()

for epoch in range(num_epochs):
  print('Epoch #' + str(epoch) + '/' + str(num_epochs))

  loss_epoch = 0

  A2Bloss_epoch = 0
  B2Aloss_epoch = 0
  DAloss_epoch = 0
  DBloss_epoch = 0

  A2B_list = []
  real_list_A = []
  B2A_list = []
  real_list_B = []
  cycle_list_A = []
  cycle_list_B = []
  
  for i, data in enumerate(train_loader):
    real_img_A = data["A"].to(device)
    real_img_B = data["B"].to(device)

    real_list_A.append(real_img_A)
    real_list_B.append(real_img_B)

    real_label = torch.ones(num_patches, dtype=torch.float).to(device)
    fake_label = torch.zeros(num_patches, dtype=torch.float).to(device)
    
    # Generators
    G_optim.zero_grad()

    out_A2B = G_A2B(real_img_A)
    out_B2A = G_B2A(real_img_B)
    B2A_list.append(out_B2A)
    A2B_list.append(out_A2B)
    pred_labelA = D_A(out_B2A)
    pred_labelB = D_B(out_A2B)

    # Train generators to generate more real looking images
    loss_G_B2A = adversarial_loss(pred_labelA, real_label)
    B2Aloss_epoch += loss_G_B2A
    loss_G_A2B = adversarial_loss(pred_labelB, real_label)
    A2Bloss_epoch += loss_G_A2B

    # Cycle Loss
    cycle_img_A = G_B2A(out_A2B)
    cycle_img_B = G_A2B(out_B2A)
    cycle_list_A.append(cycle_img_A)
    cycle_list_B.append(cycle_img_B)

    cycle_loss_A = cycle_loss(cycle_img_A, real_img_A) * lam
    cycle_loss_B = cycle_loss(cycle_img_B, real_img_B) * lam

    total_loss_G = loss_G_A2B + loss_G_B2A + cycle_loss_A + cycle_loss_B
    #total_loss_G = loss_G_B2A + cycle_loss_A + cycle_loss_B

    total_loss_G.backward()
    G_optim.step()

    # Discriminator A

    D_A_optim.zero_grad()

    # Train discriminator to accurately determine real and fake images
    real_out_A = D_A(real_img_A)
    loss_real_A = adversarial_loss(real_out_A, real_label)
    
    fake_img_A = fake_A_buffer.push_and_pop(out_B2A)
    fake_out_A = D_A(fake_img_A.detach())
    loss_fake_A = adversarial_loss(fake_out_A, fake_label)

    loss_D_A = (loss_real_A + loss_fake_A) / 2
    DAloss_epoch += loss_D_A
    loss_D_A.backward()
    D_A_optim.step()

    # Discriminator B

    D_B_optim.zero_grad()

    # Train discriminator to accurately determine real and fake images
    real_out_B = D_B(real_img_B)
    loss_real_B = adversarial_loss(real_out_B, real_label)
    
    fake_img_B = fake_B_buffer.push_and_pop(out_A2B)
    fake_out_B = D_B(fake_img_B.detach())
    loss_fake_B = adversarial_loss(fake_out_B, fake_label)

    loss_D_B = (loss_real_B + loss_fake_B) / 2
    DBloss_epoch += loss_D_B
    loss_D_B.backward()
    D_B_optim.step()

    loss_epoch += (loss_D_A.detach() + loss_D_B.detach() + total_loss_G.detach())

  print(loss_epoch)

  G_A2B_loss_traj.append(A2Bloss_epoch)
  G_B2A_loss_traj.append(B2Aloss_epoch)
  D_A_loss_traj.append(DAloss_epoch)
  D_B_loss_traj.append(DBloss_epoch)

  # Take 5 random images from each domain and plot them, their mapping, and
  # their cyclic image.

  fig, axes = plt.subplots(nrows = 5, ncols = 3, figsize=(20,20))
  fig2, axes2 = plt.subplots(nrows = 5, ncols = 3, figsize=(20,20))

  for i in range(5):
    x = random.randint(0, 1599)
    std = np.array([0.5, 0.5, 0.5])
    mean = np.array([0.5, 0.5, 0.5])

    real_image_A = real_list_A[x].detach().squeeze(0).cpu().numpy()
    A2B_img = A2B_list[x].squeeze(0).detach().cpu().permute(1, 2, 0)
    cycle_image_A = cycle_list_A[x].detach().squeeze(0).cpu().numpy()
    real_image_B = real_list_B[x].detach().squeeze(0).cpu().numpy()
    B2A_img = B2A_list[x].detach().squeeze(0).cpu().numpy()
    cycle_image_B = cycle_list_B[x].detach().squeeze(0).cpu().numpy()

    axes[i, 0].imshow(real_image_A.transpose((1, 2, 0)) * std + mean)
    axes[i, 1].imshow(A2B_img * 0.5 + 0.5)
    axes[i, 2].imshow(cycle_image_A.transpose((1, 2, 0)) * std + mean)
    axes2[i, 0].imshow(real_image_B.transpose((1, 2, 0)) * std + mean)
    axes2[i, 1].imshow(B2A_img.transpose((1, 2, 0)) * std + mean)
    axes2[i, 2].imshow(cycle_image_B.transpose((1, 2, 0)) * std + mean)

  fig.savefig(f'sample_outs/DomainReal_epoch_{epoch}.png')
  fig2.savefig(f'sample_outs/DomainCartoon_epoch_{epoch}.png')

  cur_epoch += 1

  # Save network weights
  torch.save(G_A2B.state_dict(), f'weights/G_A2B_epoch{epoch}_new.pth')
  torch.save(G_B2A.state_dict(), f'weights/G_B2A_epoch{epoch}_new.pth')
  torch.save(D_A.state_dict(), f'weights/D_A_epoch{epoch}_new.pth')
  torch.save(D_B.state_dict(), f'weights/D_B_epoch{epoch}_new.pth')

torch.save(G_A2B.state_dict(), f'weights/G_A2B_epoch{num_epochs}_new.pth')
torch.save(G_B2A.state_dict(), f'weights/G_B2A_epoch{num_epochs}_new.pth')
torch.save(D_A.state_dict(), f'weights/D_A_epoch{num_epochs}_new.pth')
torch.save(D_B.state_dict(), f'weights/D_B_epoch{num_epochs}_new.pth')

# Plot test data and save figures
fig_B_real, ax_B_real = plt.subplots(5, 5, figsize=(50,50))
fig_A_real, ax_A_real = plt.subplots(5, 5, figsize=(50,50))
fig_B_fake, ax_B_fake = plt.subplots(5, 5, figsize=(50,50))
fig_A_fake, ax_A_fake = plt.subplots(5, 5, figsize=(50,50))
fig_B_cyc, ax_B_cyc = plt.subplots(5, 5, figsize=(50,50))
fig_A_cyc, ax_A_cyc = plt.subplots(5, 5, figsize=(50,50))

j = 0
std = np.array([0.5, 0.5, 0.5])
mean = np.array([0.5, 0.5, 0.5])

for i, data in enumerate(test_loader):
  if i < 25:
    real_img_A = data["A"].to(device)
    real_img_B = data["B"].to(device)

    A2B = G_A2B(real_img_A)
    B2A = G_B2A(real_img_B)

    cycA = G_B2A(A2B)
    cycB = G_A2B(B2A)

    real_img_A = real_img_A.detach().squeeze(0).cpu().numpy()
    real_img_B = real_img_B.detach().squeeze(0).cpu().numpy()
    A2B = A2B.detach().squeeze(0).cpu().numpy()
    B2A = B2A.detach().squeeze(0).cpu().numpy()
    cycA = cycA.detach().squeeze(0).cpu().numpy()
    cycB = cycB.detach().squeeze(0).cpu().numpy()

    ax_A_real[j, i%5].imshow(real_img_A.transpose((1, 2, 0)) * std + mean)
    ax_B_fake[j, i%5].imshow(A2B.transpose((1, 2, 0)) * std + mean)
    ax_A_cyc[j, i%5].imshow(cycA.transpose((1, 2, 0)) * std + mean)
    ax_B_real[j, i%5].imshow(real_img_B.transpose((1, 2, 0)) * std + mean)
    ax_A_fake[j, i%5].imshow(B2A.transpose((1, 2, 0)) * std + mean)
    ax_B_cyc[j, i%5].imshow(cycB.transpose((1, 2, 0)) * std + mean)

    if i%5==4:
      j+=1

fig_B_real.savefig('final-images/real_B.png')
fig_A_real.savefig('final-images/real_A.png')
fig_B_fake.savefig('final-images/fake_B.png')
fig_A_fake.savefig('final-images/fake_A.png')
fig_B_cyc.savefig('final-images/cyc_B.png')
fig_A_cyc.savefig('final-images/cyc_A.png')

fig, ax = plt.subplots()
ax.plot(G_A2B_loss_traj)
fig.savefig("losses/G_A2B_Loss.png")

fig2, ax2 = plt.subplots()
ax2.plot(G_B2A_loss_traj)
fig2.savefig("losses/G_B2A_Loss.png")

fig3, ax3 = plt.subplots()
ax3.plot(D_A_loss_traj)
fig3.savefig("losses/D_A_Loss.png")

fig4, ax4 = plt.subplots()
ax4.plot(D_B_loss_traj)
fig4.savefig("losses/D_B_Loss.png")