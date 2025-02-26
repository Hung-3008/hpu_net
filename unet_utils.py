# unet_utils.py (PyTorch version)
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Architectural blocks and utility functions of the U-Net."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, n_channels, n_down_channels=None, activation_fn=nn.ReLU()):
        super(ResBlock, self).__init__()
        if n_down_channels is None:
            n_down_channels = n_channels
            
        self.activation_fn = activation_fn
        self.convs = nn.ModuleList()
        
        for i in range(3):  # Hardcoded 3 as per paper
            in_ch = in_channels if i == 0 else n_down_channels
            out_ch = n_down_channels if i < 2 else n_channels
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
            self.convs.append(conv)
            
        self.skip_conv = None
        if in_channels != n_channels:
            self.skip_conv = nn.Conv2d(in_channels, n_channels, kernel_size=1, padding=0)
            
    def forward(self, input_features):
        skip = input_features
        residual = self.activation_fn(input_features)
        
        for i, conv in enumerate(self.convs):
            residual = conv(residual)
            if i < len(self.convs) - 1:
                residual = self.activation_fn(residual)
                
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)
            
        return skip + residual

def resize_up(input_features, scale=2):
    """Nearest neighbor rescaling-operation for the input features."""
    assert scale >= 1
    return F.interpolate(input_features, scale_factor=scale, 
                        mode='nearest', align_corners=None)

def resize_down(input_features, scale=2):
    """Average pooling rescaling-operation for the input features."""
    assert scale >= 1
    return F.avg_pool2d(input_features, kernel_size=scale, stride=scale)