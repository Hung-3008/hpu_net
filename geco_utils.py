import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MovingAverage(nn.Module):
    """
    A simple moving average module.

    The module updates an internal state with the rule:
      ma = decay * ma + (1 - decay) * new_value

    If differentiable is False, the new input is detached.
    """
    def __init__(self, decay, differentiable=False, name='moving_average'):
        super(MovingAverage, self).__init__()
        self.decay = decay
        self.differentiable = differentiable
        self.ma = None  # internal moving average state

    def forward(self, x):
        if not self.differentiable:
            x = x.detach()
        if self.ma is None:
            self.ma = x
        else:
            self.ma = self.decay * self.ma + (1.0 - self.decay) * x
        return self.ma


class LagrangeMultiplier(nn.Module):
    """
    A simple Lagrange multiplier module.

    The multiplier is implemented as a learnable parameter. The parameter
    is multiplied by the given rate to scale the gradient update.
    """
    def __init__(self, rate=1e-2, name='lagrange_multiplier'):
        super(LagrangeMultiplier, self).__init__()
        self.rate = rate
        # Initialize with ones (scalar); shape can be adapted as needed.
        self.lagrange = nn.Parameter(torch.ones(1))

    def forward(self, ma_constraint):
        # Return the scaled Lagrange multiplier.
        return self.lagrange * self.rate


def _sample_gumbel(shape, eps=1e-20):
    """
    Sample noise from a standard Gumbel distribution.
    
    This is done by applying the inverse transform:
      G = -log(-log(U + eps) + eps)
    where U ~ Uniform(0,1).
    """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def _topk_mask(score, k):
    """
    Returns a mask with ones at the positions of the top-k values in `score`
    and zeros elsewhere.
    
    Args:
      score: A 1D tensor.
      k: Number of top elements to keep.
    """
    # Get top-k indices
    _, indices = torch.topk(score, k)
    mask = torch.zeros_like(score)
    mask[indices] = 1.0
    return mask


def ce_loss(logits, labels, mask=None, top_k_percentage=None, deterministic=False):
    """
    Computes the cross-entropy loss with an optional top-k masking.

    Args:
      logits: Tensor of shape (B, H, W, num_classes).
      labels: Tensor of shape (B, H, W, num_classes) (one-hot encoded).
      mask: None or tensor of shape (B, H, W). If None, all pixels are used.
      top_k_percentage: None or a float in (0., 1.]. If provided, only the
          top k-percent pixels (by normalized loss) contribute.
      deterministic: Boolean; if True the top-k mask is computed deterministically.
    
    Returns:
      A dict with keys:
        'mean': the mean loss per pixel,
        'sum': the per-batch sum of losses (averaged over batch),
        'mask': the final loss mask.
    """
    num_classes = logits.shape[-1]
    # Reshape logits and labels to 2D: (N, num_classes) where N = B*H*W.
    y_flat = logits.view(-1, num_classes)
    t_flat = labels.view(-1, num_classes)
    
    if mask is None:
        mask = torch.ones(t_flat.shape[0], device=logits.device)
    else:
        expected_shape = labels.shape[:-1]
        assert mask.shape == expected_shape, \
            f'The loss mask shape differs from the target shape: {mask.shape} vs. {expected_shape}.'
        mask = mask.view(-1)

    n_pixels = y_flat.shape[0]
    # Compute the per-pixel cross-entropy loss.
    log_probs = F.log_softmax(y_flat, dim=1)
    xe = -torch.sum(t_flat * log_probs, dim=1)  # shape (N,)

    if top_k_percentage is not None:
        assert 0.0 < top_k_percentage <= 1.0, "top_k_percentage must be in (0, 1]."
        # Compute number of pixels to keep.
        k_pixels = int(torch.floor(torch.tensor(n_pixels * top_k_percentage)).item())

        stopgrad_xe = xe.detach()
        norm_xe = stopgrad_xe / torch.sum(stopgrad_xe)
        if deterministic:
            score = torch.log(norm_xe)
        else:
            score = torch.log(norm_xe) + _sample_gumbel(norm_xe.shape)
        # Adding log(mask) will set positions where mask==0 to -inf.
        score = score + torch.log(mask)
        topk_mask = _topk_mask(score, k_pixels)
        mask = mask * topk_mask

    # Reshape xe and mask back to (B, -1)
    batch_size = logits.shape[0]
    xe = xe.view(batch_size, -1)
    mask = mask.view(batch_size, -1)
    ce_sum_per_instance = torch.sum(mask * xe, dim=1)
    ce_sum = torch.mean(ce_sum_per_instance)
    ce_mean = torch.sum(mask * xe) / torch.sum(mask)

    return {'mean': ce_mean, 'sum': ce_sum, 'mask': mask}