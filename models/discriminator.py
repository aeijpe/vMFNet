import  torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F



class DiscriminatorC(nn.Module):
  def __init__(self, input_channels=64):
    super(DiscriminatorC, self).__init__()

    self.conv_blocks = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1), # 32, 62, 62
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=1), # 64, 29, 29
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=1), # 128, 13, 13
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=1), # 256, 5, 5
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=0), # 256, 1, 1
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
      )
    

  def forward(self, x):
    out = self.conv_blocks(x)
    out = out.squeeze()
    return out
    


class DiscriminatorD(nn.Module):
  def __init__(self, input_channels=1):
    super(DiscriminatorD, self).__init__()
    # input = 1 X 256 x 256

    self.conv_blocks = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1), # 64, 128, 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128, 64, 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), #  256, 32, 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 512, 16, 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), # 1024, 8, 8
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1), # 2048, 4, 4
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2048, 1, kernel_size=1, stride=1, padding=0), # 1, 4, 4

        )
      


  def forward(self, x):
    
    out = self.conv_blocks(x)
    # reshape out from [B, 1, 4, 4] TO [B, 1, 16]
    out = out.view(out.size(0), 1, -1)
    return out