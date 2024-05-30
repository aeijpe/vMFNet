import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time

from models.unet_model import *

class Encoder(nn.Module):
    def __init__(self, num_output_channels, type="unet", norm="Batch"):
        super(Encoder, self).__init__()
        """
        Build an encoder to extract anatomical information from the image.
        """
        self.num_output_channels = num_output_channels
        if type=="unet":
            self.unet = UNetNorm(n_classes=self.num_output_channels, norm=norm)
        else: 
            print("making a res unet encoder!!")
            self.unet = CustomUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=self.num_output_channels,
                channels=(64, 128, 256, 512, 1024),
                strides=(2, 2, 2, 2),
                num_res_units=1,
            )

    def forward(self, x):
        out = self.unet(x)
        return out
    
  
