import  torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F



class Discriminator(nn.Module):
  def __init__(self, input_channels=12):
    super(Discriminator, self).__init__()
    self.conv_blocks = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=1), # 32, 62, 62
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=1), # 64, 29, 29
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=1), # 128, 13, 13
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=1), # 256, 5, 5
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=0), # 256, 1, 1
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
      
    self.fc = nn.Sequential(
        nn.Linear(256, 16),  # Adjust the size here based on the output of the last conv layer
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(16, 1), 
        nn.Sigmoid() # not
    )

  def forward(self, x):
    out = self.conv_blocks(x)
    #print("out conv block", out.shape)
    out = out.squeeze()
    out = self.fc(out)
    return out