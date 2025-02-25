import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    """PyTorch implementation of moving average module."""
    
    def __init__(self, decay, differentiable=False):
        super(MovingAverage, self).__init__()
        self.decay = decay
        self.differentiable = differentiable
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('updates', torch.tensor(0.0))

    def forward(self, inputs):
        if not self.differentiable:
            inputs = inputs.detach()
            
        if self.updates == 0:
            self.running_mean = inputs
        else:
            self.running_mean = self.running_mean * self.decay + inputs * (1 - self.decay)
            
        self.updates += 1
        return self.running_mean


class LagrangeMultiplier(nn.Module):
    """PyTorch implementation of Lagrange multiplier module."""
    
    def __init__(self, rate=1e-2, shape=None):
        super(LagrangeMultiplier, self).__init__()
        self.rate = rate
        if shape is None:
            shape = [1]
        self.lagrange_multiplier = nn.Parameter(torch.ones(shape))

    def forward(self, ma_constraint):
        return self.lagrange_multiplier * ma_constraint


def _sample_gumbel(shape, eps=1e-20, device='cuda'):
    """Sample from Gumbel distribution."""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def _topk_mask(score, k):
    """Returns a mask for the top-k elements in score."""
    values, indices = torch.topk(score.flatten(), k=k)
    mask = torch.zeros_like(score.flatten())
    mask[indices] = 1.0
    return mask.reshape(score.shape)


def ce_loss(logits, labels, mask=None, top_k_percentage=None, deterministic=False):
    """PyTorch implementation of cross-entropy loss with optional top-k masking."""
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[-1]
    
    # Flatten the inputs
    y_flat = logits.reshape(-1, num_classes)
    t_flat = labels.reshape(-1, num_classes)
    
    if mask is None:
        mask = torch.ones(t_flat.shape[0], device=logits.device)
    else:
        assert mask.shape[:3] == labels.shape[:3], \
            f'The loss mask shape differs from the target shape: {mask.shape} vs {labels.shape[:3]}'
        mask = mask.reshape(-1)

    n_pixels_in_batch = y_flat.shape[0]
    xe = F.cross_entropy(y_flat, t_flat.argmax(dim=1), reduction='none')

    if top_k_percentage is not None:
        assert 0.0 < top_k_percentage <= 1.0
        k_pixels = int(n_pixels_in_batch * top_k_percentage)

        with torch.no_grad():
            norm_xe = xe / xe.sum()

        if deterministic:
            score = torch.log(norm_xe)
        else:
            score = torch.log(norm_xe) + _sample_gumbel(
                norm_xe.shape, device=logits.device)

        score = score + torch.log(mask)
        top_k_mask = _topk_mask(score, k_pixels)
        mask = mask * top_k_mask

    # Calculate batch statistics
    xe = xe.reshape(batch_size, -1)
    mask = mask.reshape(batch_size, -1)
    
    ce_sum_per_instance = (mask * xe).sum(dim=1)
    ce_sum = ce_sum_per_instance.mean()
    ce_mean = (mask * xe).sum() / mask.sum()

    return {
        'mean': ce_mean,
        'sum': ce_sum,
        'mask': mask
    }