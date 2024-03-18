import torch.nn as nn
import torch.nn.functional as F
from models.unet_parts import *
from models.blocks import *
import torch


class Segmentor(nn.Module):
    def __init__(self, num_classes, layer=8, bilinear=True):
        super(Segmentor, self).__init__()

        self.anatomy_out_channels = 0
        self.num_classes = num_classes
        self.layer = layer

        
        input_channels =  12 
        out_channels = 64

        self.conv1 = DoubleConv(input_channels, out_channels)
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 64)

        self.outc = OutConv(64, self.num_classes)

    def forward(self, content, features=None):
        
        out = self.conv1(content)
        out = self.up4(out)
        out = self.conv2(out)

        out = self.outc(out)
        out = F.softmax(out, dim=1)
        return out