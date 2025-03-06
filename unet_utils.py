import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """A pre-activated residual block."""
    def __init__(self, in_channels, out_channels, down_channels=None, 
                 activation=nn.ReLU, convs_per_block=3):
        super().__init__()
        self.activation = activation()
        down_channels = down_channels or out_channels
        
        self.convs = nn.ModuleList()
        for i in range(convs_per_block):
            self.convs.append(
                nn.Conv2d(in_channels if i==0 else down_channels,
                          down_channels, kernel_size=3, padding=1)
            )
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.residual = nn.Conv2d(down_channels, out_channels, kernel_size=1) if down_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.activation(x)
        
        for conv in self.convs[:-1]:
            x = self.activation(conv(x))
        x = self.convs[-1](x)
        
        return self.shortcut(residual) + self.residual(x)

def resize_up(x, scale=2):
    """Nearest neighbor upsampling."""
    return F.interpolate(x, scale_factor=scale, mode='nearest')

def resize_down(x, scale=2):
    """Average pooling downsampling."""
    return F.avg_pool2d(x, kernel_size=scale, stride=scale)