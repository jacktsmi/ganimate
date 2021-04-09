import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Encoder
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=1,),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # Transformer (Residual Blocks)
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),

            # Decoder
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=2,),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=(7, 7), stride=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        out = self.main(x)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        self.main = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=(4, 4), stride=2),
                        nn.InstanceNorm2d(64),
                        nn.LeakyReLU(negative_slope = 0.2),
                        
                        nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2),
                        nn.InstanceNorm2d(128),
                        nn.LeakyReLU(negative_slope = 0.2),

                        nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2),
                        nn.InstanceNorm2d(256),
                        nn.LeakyReLU(negative_slope = 0.2),

                        nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2),
                        nn.InstanceNorm2d(512),
                        nn.LeakyReLU(negative_slope = 0.2),

                        nn.Conv2d(512, 1, kernel_size=(4, 4), stride=1),
                        nn.Sigmoid()
                    )
    def forward(self, x):
        out = self.main(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
    
    def forward(self, x):
        residual = x
        out1 = nn.InstanceNorm2d(F.relu(self.conv1(x), inplace=True))
        out2 = nn.InstanceNorm2d(F.relu(self.conv2(out1), inplace=True))
        out = out2 + residual
        out = out.view(out.size(0), -1)

        return out

# Initialize the weights of the generator
def weight_init_g(layer):
    if type(layer) == nn.ConvTranspose2d:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif type(layer) == nn.BatchNorm2d:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0.0)
    # end if
# end weight_init_generator

# Initialize the weights of the discriminator
def weight_init_d(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif type(layer) == nn.BatchNorm2d:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0.0)
    # end if
# end weight_init_generator