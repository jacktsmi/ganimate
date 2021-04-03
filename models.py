import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.enc1 = nn.InstanceNorm2d(nn.ReLU(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1)))
        self.enc2 = nn.InstanceNorm2d(nn.ReLU(nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2)))
        self.enc3 = nn.InstanceNorm2d(nn.ReLU(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)))
        # Transformer (Residual Blocks)
        self.res1 = ResidualBlock()
        self.res2 = ResidualBlock()
        self.res3 = ResidualBlock()
        self.res4 = ResidualBlock()
        self.res5 = ResidualBlock()
        self.res6 = ResidualBlock()
        # Decoder
        self.dec1 = nn.InstanceNorm2d(nn.ReLU(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2)))
        self.dec2 = nn.InstanceNorm2d(nn.ReLU(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2)))
        self.dec3 = nn.InstanceNorm2d(nn.ReLU(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1)))
    
    def forward(self, x):
        out = self.enc1(x)
        out = self.enc2(out)
        out = self.enc3(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.dec1(out)
        out = self.dec2(out)
        out = self.dec3(out)

        return out

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
    
    def forward(self, x):
        residual = x
        out1 = nn.InstanceNorm2d(nn.ReLU(self.conv1(x)))
        out2 = nn.InstanceNorm2d(nn.ReLU(self.conv2(out1)))
        out = out2 + residual
        out = out.view(out.size(0), -1)

        return out