

import torch
import torch.nn as nn
from torch.nn import functional


class attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_conv = nn.Sequential(
            nn.ReLU(),
            nn.Sigmoid()
            )
    def forward(self,x):
        return self.attention_conv(x)

class SingleConv(nn.Module):
    
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.single_conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.single_conv(x)

class SimpleConv(nn.Module):
    
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.simple_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.simple_conv(x)
    
class DoubleConv(nn.Module):
    
    def __init__(self, out_chan):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.double_conv(x)

class Dec_attention1(nn.Module):

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, in_chan, kernel_size=2, stride=2)
        self.dcatt = nn.Sequential(
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.down_channel = nn.Conv2d(1024, out_chan, kernel_size=3, padding=1)
        self.conv = DoubleConv(out_chan)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x3 = torch.cat([x1, x2], dim=1)
        x4 = self.down_channel(x3)
        x = self.dcatt(x4)
        return self.conv(x)

class Dec_attention2(nn.Module):

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, in_chan, kernel_size=2, stride=2)
        self.dcatt = nn.Sequential(
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.down_channel = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.conv = DoubleConv(out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x3 = self.down_channel(x1)
        x4 = torch.cat([x3, x2], dim=1)
        x5 = self.down_channel(x4)
        x = self.dcatt(x5)
        return self.conv(x)

class Dec_attention3(nn.Module):

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, in_chan, kernel_size=2, stride=2)
        self.dcatt = nn.Sequential(
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.down_channel = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x3 = self.down_channel(x1)
        x4 = torch.cat([x3, x2], dim=1)
        x5 = self.down_channel(x4)
        x = self.dcatt(x5)
        return x

def up_sample2d(x, t, mode="bilinear"):
    
    return functional.interpolate(x, t.size()[2:], mode=mode, align_corners=False)