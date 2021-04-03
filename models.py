import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.conv1 = nn.BatchNorm2d(nn.ReLU(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1)))
        self.conv2 = nn.BatchNorm2d(nn.ReLU(nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2)))
        self.conv3 = nn.BatchNorm2d(nn.ReLU(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)))
        # Transformer (Residual Blocks)

        # Decoder