import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    """A moving average module that tracks the exponential moving average."""
    def __init__(self, decay, differentiable=False):
        super().__init__()
        self.decay = decay
        self.differentiable = differentiable
        self.register_buffer('average', None)

    def forward(self, inputs):
        if not self.differentiable:
            inputs = inputs.detach()
        
        if self.average is None:
            average = inputs.detach().clone()
        else:
            average_prev = self.average.detach()
            average = self.decay * average_prev + (1 - self.decay) * inputs
        
        # Update buffer with detached average
        self.average = average.detach()
        return average

class LagrangeMultiplier(nn.Module):
    """A Lagrange multiplier module that scales gradients by a rate."""
    def __init__(self, rate=1e-2):
        super().__init__()
        self.rate = rate
        self.lagmul = nn.Parameter(torch.ones(1))

    def forward(self, ma_constraint):
        return self.rate * self.lagmul

def _sample_gumbel(shape):
    """Sample Gumbel noise for top-k sampling."""
    uniform = torch.rand(shape).float()
    return -torch.log(-torch.log(uniform + 1e-20) + 1e-20)

def _topk_mask(score, k):
    """Create a binary mask for the top-k elements in score."""
    _, topk_indices = torch.topk(score, k, largest=True)
    topk_mask = torch.zeros_like(score)
    topk_mask.scatter_(0, topk_indices, 1)
    return topk_mask

def ce_loss(logits, labels, mask=None, top_k_percentage=None, deterministic=False):
    """Computes the cross-entropy loss with optional masking and top-k selection.

    Args:
        logits: Tensor of shape [B, C, H, W].
        labels: Tensor of shape [B, C, H, W] or [B, 1, H, W].
        mask: None or tensor of shape [B, 1, H, W] or [B, H, W].
        top_k_percentage: None or float in (0., 1.] for top-k selection.
        deterministic: Boolean indicating deterministic top-k selection.

    Returns:
        Dictionary with 'mean' (mean masked loss), 'sum' (batch-averaged sum), and 'mask'.
    """
    num_classes = logits.shape[1]
    # Convert labels to one-hot if needed
    if labels.shape[1] == 1:
        labels = torch.nn.functional.one_hot(labels.squeeze(1).long(), num_classes).permute(0, 3, 1, 2).float()

    batch_size = labels.shape[0]

    # Flatten logits and labels
    y_flat = logits.view(batch_size, num_classes, -1).transpose(1, 2).reshape(-1, num_classes)
    t_flat = labels.view(batch_size, num_classes, -1).transpose(1, 2).reshape(-1, num_classes)

    # Handle mask
    if mask is None:
        mask = torch.ones(y_flat.size(0), device=logits.device)
    else:
        mask = mask.view(-1).float()  # Reshape to [B*H*W,]

    # Compute cross-entropy
    xe = - (t_flat * F.log_softmax(y_flat, dim=1)).sum(dim=1)
    n_pixels_in_batch = y_flat.size(0)

    # Top-k selection
    if top_k_percentage is not None:
        assert 0.0 < top_k_percentage <= 1.0, "top_k_percentage must be in (0, 1]"
        k_pixels = int(n_pixels_in_batch * top_k_percentage)

        with torch.no_grad():
            stopgrad_xe = xe.detach()
            norm_xe = stopgrad_xe / stopgrad_xe.sum()

        if deterministic:
            score = torch.log(norm_xe)
        else:
            gumbel = _sample_gumbel(norm_xe.shape).to(xe.device)
            score = torch.log(norm_xe) + gumbel

        score = score + torch.log(mask + 1e-20)  # Apply mask in log-space
        top_k_mask = _topk_mask(score, k_pixels)
        mask = mask * top_k_mask

    # Reshape for batch aggregation
    xe = xe.view(batch_size, -1)
    mask = mask.view(batch_size, -1).float()

    # Compute loss metrics
    ce_sum_per_instance = (mask * xe).sum(dim=1)
    ce_sum = ce_sum_per_instance.mean()
    ce_mean = (mask * xe).sum() / (mask.sum() + 1e-20)

    return {'mean': ce_mean, 'sum': ce_sum, 'mask': mask}