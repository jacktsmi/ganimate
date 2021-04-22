import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Initial convolution block
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Downsampling
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # Residual blocks
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            # Upsampling
            nn.Upsample(scale_factor = 2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor = 2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )
    
    def forward(self, x):
        out = self.main(x)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 5, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 5, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 5, stride=2, padding=0),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 5, padding=0)
        )
    def forward(self, x):
        out = self.main(x)
        out2 = torch.flatten(out)
        return out2

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.res = nn.Sequential(nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels),
                                 nn.ReLU(inplace=True),
                                 nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels))
    
    def forward(self, x):
        return x + self.res(x)

# Initialize the weights of the generator
def weight_init_g(layer):
    if type(layer) == nn.ConvTranspose2d:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif type(layer) == nn.InstanceNorm2d:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0.0)
    # end if
# end weight_init_generator

# Initialize the weights of the discriminator
def weight_init_d(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif type(layer) == nn.InstanceNorm2d:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0.0)
    # end if
# end weight_init_generator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

class ImageBuffer():
  def __init__(self):
    self.data = []
    self.max_size = 50
  
  def push_and_pop(self, data):
    to_return = []
    for el in data.data:
      el = torch.unsqueeze(el, 0)
      if len(self.data) < self.max_size:
          self.data.append(el)
          to_return.append(el)
      else:
          if random.uniform(0, 1) > 0.5:
              i = random.randint(0, self.max_size - 1)
              to_return.append(self.data[i].clone())
              self.data[i] = el
          else:
              to_return.append(el)
    return torch.cat(to_return)