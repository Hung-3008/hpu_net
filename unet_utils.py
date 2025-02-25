from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


def res_block(input_features: torch.Tensor,
              n_channels: int,
              n_down_channels: Optional[int] = None,
              activation_fn: Callable = F.relu,
              convs_per_block: int = 3) -> torch.Tensor:
    """A pre-activated residual block.

    Args:
        input_features: A tensor of shape (b, c, h, w)
        n_channels: Number of output channels
        n_down_channels: Number of intermediate channels
        activation_fn: Callable activation function
        convs_per_block: Number of convolutional layers
    Returns:
        A tensor of shape (b, c, h, w)
    """
    # Pre-activate the inputs
    skip = input_features
    residual = activation_fn(input_features)

    # Set the number of intermediate channels that we compress to
    if n_down_channels is None:
        n_down_channels = n_channels

    for c in range(convs_per_block):
        residual = nn.Conv2d(
            in_channels=residual.size(1),
            out_channels=n_down_channels,
            kernel_size=3,
            padding=1,
            bias=True)(residual)
        if c < convs_per_block - 1:
            residual = activation_fn(residual)

    incoming_channels = input_features.size(1)
    if incoming_channels != n_channels:
        skip = nn.Conv2d(
            in_channels=incoming_channels,
            out_channels=n_channels,
            kernel_size=1,
            padding=0,
            bias=True)(skip)
    if n_down_channels != n_channels:
        residual = nn.Conv2d(
            in_channels=n_down_channels,
            out_channels=n_channels,
            kernel_size=1,
            padding=0,
            bias=True)(residual)
    return skip + residual


def resize_up(input_features: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Nearest neighbor rescaling-operation for the input features.

    Args:
        input_features: A tensor of shape (b, c, h, w)
        scale: Scaling factor
    Returns:
        A tensor of shape (b, c, scale * h, scale * w)
    """
    assert scale >= 1
    _, _, size_x, size_y = input_features.shape
    new_size_x = int(round(size_x * scale))
    new_size_y = int(round(size_y * scale))
    return F.interpolate(
        input_features,
        size=(new_size_x, new_size_y),
        mode='nearest',
        align_corners=None)


def resize_down(input_features: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Average pooling rescaling-operation for the input features.

    Args:
        input_features: A tensor of shape (b, c, h, w)
        scale: Scaling factor
    Returns:
        A tensor of shape (b, c, h / scale, w / scale)
    """
    assert scale >= 1
    return F.avg_pool2d(
        input_features,
        kernel_size=scale,
        stride=scale,
        padding=0)