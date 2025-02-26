# geco_utils.py (PyTorch version)
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

"""Utility Functions for the GECO-objective.

(GECO is described in `Taming VAEs`, see https://arxiv.org/abs/1810.00597).
"""

import numpy as np
import torch
import torch.nn as nn

class MovingAverage(nn.Module):
    """A PyTorch implementation of moving average with differentiation control."""
    
    def __init__(self, decay, local=True, differentiable=False, name='moving_average'):
        super(MovingAverage, self).__init__()
        self.decay = decay
        self.local = local
        self.differentiable = differentiable
        self.register_buffer('moving_avg', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0.0))
        
    def forward(self, inputs):
        if not self.differentiable:
            inputs = inputs.detach()
            
        if self.training:
            with torch.no_grad():
                if self.count == 0:
                    self.moving_avg = inputs.mean()
                else:
                    self.moving_avg = (self.decay * self.moving_avg + 
                                     (1 - self.decay) * inputs.mean())
                self.count += 1
        return self.moving_avg.expand_as(inputs)

class LagrangeMultiplier(nn.Module):
    """A PyTorch implementation of Lagrange multiplier."""
    
    def __init__(self, rate=1e-2, name='lagrange_multiplier'):
        super(LagrangeMultiplier, self).__init__()
        self.rate = rate
        self.lagmul = nn.Parameter(torch.ones(1))
        
    def forward(self, ma_constraint):
        return self.lagmul * ma_constraint

def _sample_gumbel(shape, eps=1e-20, device='cpu'):
    """Transforms a uniform random variable to be standard Gumbel distributed."""
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)

def _topk_mask(score, k):
    """Returns a mask for the top-k elements in score."""
    _, indices = torch.topk(score, k=k)
    mask = torch.zeros_like(score)
    mask.scatter_(0, indices, 1.0)
    return mask

def ce_loss(logits, labels, mask=None, top_k_percentage=None, deterministic=False):
    """Computes the cross-entropy loss.

    Optionally a mask and a top-k percentage for the used pixels can be specified.
    """
    num_classes = logits.shape[-1]
    device = logits.device
    
    y_flat = logits.reshape(-1, num_classes)
    t_flat = labels.reshape(-1, num_classes)
    
    if mask is None:
        mask = torch.ones(t_flat.shape[0], device=device)
    else:
        assert mask.shape[:3] == labels.shape[:3], \
            f'The loss mask shape differs from the target shape: {mask.shape} vs. {labels.shape[:3]}'
        mask = mask.reshape(-1)

    n_pixels_in_batch = y_flat.shape[0]
    xe = nn.functional.cross_entropy(y_flat, t_flat.argmax(dim=-1), reduction='none')

    if top_k_percentage is not None:
        assert 0.0 < top_k_percentage <= 1.0
        k_pixels = int(np.floor(n_pixels_in_batch * top_k_percentage))

        stopgrad_xe = xe.detach()
        norm_xe = stopgrad_xe / stopgrad_xe.sum()

        if deterministic:
            score = torch.log(norm_xe)
        else:
            score = torch.log(norm_xe) + _sample_gumbel(norm_xe.shape, device=device)

        score = score + torch.log(mask)
        top_k_mask = _topk_mask(score, k_pixels)
        mask = mask * top_k_mask

    batch_size = labels.shape[0]
    xe = xe.reshape(batch_size, -1)
    mask = mask.reshape(batch_size, -1)
    
    ce_sum_per_instance = (mask * xe).sum(dim=1)
    ce_sum = ce_sum_per_instance.mean(dim=0)
    ce_mean = (mask * xe).sum() / mask.sum()

    return {'mean': ce_mean, 'sum': ce_sum, 'mask': mask}