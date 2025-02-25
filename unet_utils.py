import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    A pre-activated residual block.

    This block applies an activation to the input, passes it through a series
    of convolutional layers (using kernel size 3 with padding to preserve spatial dimensions),
    and then adds a skip connection. If the input number of channels differs from the desired
    output channels, a 1x1 convolution is applied to the skip. Similarly, if the intermediate
    (down) channels differ from the output channels, a 1x1 convolution is applied to the residual.

    Args:
      in_channels: Number of input channels.
      out_channels: Desired number of output channels.
      mid_channels: Number of intermediate channels; if None, set to out_channels.
      convs_per_block: Number of convolutional layers in the block.
      activation_fn: Activation function to use (default: F.relu).
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, convs_per_block=3, activation_fn=F.relu):
        super(ResBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.activation_fn = activation_fn

        # First convolution: in_channels -> mid_channels, then convs_per_block-1 convs: mid_channels -> mid_channels.
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1))
        for _ in range(1, convs_per_block):
            self.convs.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))

        # If the input channel count differs from out_channels, adjust the skip connection.
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        # If the intermediate channel count differs from out_channels, adjust the residual.
        self.res_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1) if mid_channels != out_channels else None

    def forward(self, x):
        skip = x
        out = self.activation_fn(x)
        for i, conv in enumerate(self.convs):
            out = conv(out)
            if i < len(self.convs) - 1:
                out = self.activation_fn(out)
        if self.res_conv is not None:
            out = self.res_conv(out)
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)
        return skip + out


def resize_up(input_features, scale=2):
    """
    Nearest neighbor upsampling operation.

    Args:
      input_features: A tensor of shape (B, C, H, W).
      scale: Upsampling factor.
    Returns:
      A tensor of shape (B, C, scale * H, scale * W).
    """
    assert scale >= 1, "Scale must be >= 1"
    # For nearest neighbor, F.interpolate is equivalent to tf.image.resize(..., method='NEAREST_NEIGHBOR').
    return F.interpolate(input_features, scale_factor=scale, mode='nearest')


def resize_down(input_features, scale=2):
    """
    Downsampling operation via average pooling.

    Args:
      input_features: A tensor of shape (B, C, H, W).
      scale: Downsampling factor.
    Returns:
      A tensor of shape (B, C, H/scale, W/scale).
    """
    assert scale >= 1, "Scale must be >= 1"
    # F.avg_pool2d with kernel_size and stride equal to scale mimics tf.nn.avg_pool2d with ksize (1, scale, scale, 1).
    return F.avg_pool2d(input_features, kernel_size=scale, stride=scale)