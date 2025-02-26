# model.py (PyTorch version)
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

"""Open Source Version of the Hierarchical Probabilistic U-Net."""

import torch
import torch.nn as nn
import torch.distributions as tfd
from unet_utils import ResBlock, resize_up, resize_down
from geco_utils import MovingAverage, LagrangeMultiplier, ce_loss

class HierarchicalCore(nn.Module):
    def __init__(self, latent_dims, channels_per_block, input_channels=1, down_channels_per_block=None,
                 activation_fn=nn.ReLU(), initializers=None, regularizers=None,
                 convs_per_block=3, blocks_per_level=3, name='HierarchicalDecoderDist'):
        super(HierarchicalCore, self).__init__()
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.activation_fn = activation_fn
        self.convs_per_block = convs_per_block
        self.blocks_per_level = blocks_per_level
        self.input_channels = input_channels
        
        if down_channels_per_block is None:
            self.down_channels_per_block = channels_per_block
        else:
            self.down_channels_per_block = down_channels_per_block
            
        self.encoder_blocks = nn.ModuleList()
        for level, (n_ch, n_down_ch) in enumerate(zip(channels_per_block, self.down_channels_per_block)):
            level_blocks = nn.ModuleList()
            in_ch = self.input_channels if level == 0 else channels_per_block[level-1]
            for _ in range(blocks_per_level):
                block = ResBlock(in_ch, n_ch, n_down_ch, activation_fn)
                level_blocks.append(block)
                in_ch = n_ch
            self.encoder_blocks.append(level_blocks)
            
        # Decoder blocks (full 4 levels to reach 64x64)
        self.decoder_blocks = nn.ModuleList()
        for level in range(len(latent_dims)):  # All 4 levels
            if level == 0:
                in_ch = latent_dims[0] + channels_per_block[-1] + channels_per_block[-2]
            else:
                in_ch = (latent_dims[level] + 
                        self.channels_per_block[::-1][level] + 
                        self.channels_per_block[::-1][level + 1])
            level_blocks = nn.ModuleList()
            for i in range(blocks_per_level):
                out_ch = (self.channels_per_block[::-1][level + 1] if i == blocks_per_level - 1 
                         else in_ch // 2)
                block = ResBlock(in_ch, out_ch, out_ch // 2, activation_fn)
                level_blocks.append(block)
                in_ch = out_ch
            self.decoder_blocks.append(level_blocks)
            
        self.latent_projections = nn.ModuleList()
        for latent_dim in latent_dims:
            self.latent_projections.append(
                nn.Conv2d(channels_per_block[-1], 2 * latent_dim, kernel_size=1, padding=0)
            )

    def forward(self, inputs, mean=False, z_q=None):
        encoder_features = inputs
        encoder_outputs = []
        num_levels = len(self.channels_per_block)
        num_latent_levels = len(self.latent_dims)
        
        if isinstance(mean, bool):
            mean = [mean] * num_latent_levels
            
        distributions = []
        used_latents = []
        
        for level, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                encoder_features = block(encoder_features)
            encoder_outputs.append(encoder_features)
            if level != num_levels - 1:
                encoder_features = resize_down(encoder_features, scale=2)
                
        decoder_features = encoder_outputs[-1]  # [32, 192, 8, 8]
        for level in range(num_latent_levels):  # All 4 levels
            mu_logsigma = self.latent_projections[level](decoder_features)
            latent_dim = self.latent_dims[level]
            mu = mu_logsigma[:, :latent_dim, :, :]
            logsigma = mu_logsigma[:, latent_dim:, :, :]
            dist = tfd.Normal(loc=mu, scale=torch.exp(logsigma))
            distributions.append(dist)
            
            if z_q is not None:
                z = z_q[level]
            elif mean[level]:
                z = dist.loc
            else:
                z = dist.sample()
            used_latents.append(z)
            
            decoder_output_lo = torch.cat([z, decoder_features], dim=1)
            decoder_output_hi = resize_up(decoder_output_lo, scale=2)
            decoder_features = torch.cat(
                [decoder_output_hi, encoder_outputs[::-1][level + 1]], dim=1)
            
            for block in self.decoder_blocks[level]:
                decoder_features = block(decoder_features)
                
        return {
            'decoder_features': decoder_features,  # [32, 24, 64, 64]
            'encoder_features': encoder_outputs,
            'distributions': distributions,
            'used_latents': used_latents
        }

class StitchingDecoder(nn.Module):
    def __init__(self, latent_dims, channels_per_block, num_classes,
                 down_channels_per_block=None, activation_fn=nn.ReLU(),
                 initializers=None, regularizers=None, convs_per_block=3,
                 blocks_per_level=3, name='StitchingDecoder'):
        super(StitchingDecoder, self).__init__()
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.num_classes = num_classes
        self.activation_fn = activation_fn
        self.convs_per_block = convs_per_block
        self.blocks_per_level = blocks_per_level
        
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block
        self.down_channels_per_block = down_channels_per_block
        
        self.decoder_blocks = nn.ModuleList()
        num_latents = len(latent_dims)
        start_level = num_latents  # Start at level 4 (after 4 latent levels)
        prev_channels = channels_per_block[::-1][num_latents - 1]  # 24 from HierarchicalCore
        for level in range(start_level, start_level + 1):  # Only 1 level to 128x128
            in_ch = prev_channels + channels_per_block[::-1][level]
            block = nn.ModuleList([
                ResBlock(in_ch, channels_per_block[::-1][level],
                        down_channels_per_block[::-1][level], activation_fn)
                for _ in range(blocks_per_level)
            ])
            self.decoder_blocks.append(block)
            prev_channels = channels_per_block[::-1][level]
            
        self.final_conv = nn.Conv2d(channels_per_block[0], num_classes,
                                  kernel_size=1, padding=0)

    def forward(self, encoder_features, decoder_features):
        num_latents = len(self.latent_dims)
        start_level = num_latents
        
        for level, blocks in enumerate(self.decoder_blocks, start=start_level):
            decoder_features = resize_up(decoder_features, scale=2)
            decoder_features = torch.cat(
                [decoder_features, encoder_features[::-1][level]], dim=1)
            for block in blocks:
                decoder_features = block(decoder_features)
                
        return self.final_conv(decoder_features)

class HierarchicalProbUNet(nn.Module):
    """A Hierarchical Probabilistic U-Net."""
    
    def __init__(self, latent_dims=(1, 1, 1, 1), channels_per_block=None,
                 num_classes=2, down_channels_per_block=None,
                 activation_fn=nn.ReLU(), initializers=None,
                 regularizers=None, convs_per_block=3, blocks_per_level=3,
                 loss_kwargs=None, name='HPUNet'):
        super(HierarchicalProbUNet, self).__init__()
        base_channels = 24
        default_channels_per_block = (
            base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels,
            8 * base_channels, 8 * base_channels, 8 * base_channels, 8 * base_channels
        )
        if channels_per_block is None:
            channels_per_block = default_channels_per_block
        if down_channels_per_block is None:
            down_channels_per_block = tuple([i / 2 for i in default_channels_per_block])
            
        if loss_kwargs is None:
            self.loss_kwargs = {
                'type': 'geco',
                'top_k_percentage': 0.02,
                'deterministic_top_k': False,
                'kappa': 0.05,
                'decay': 0.99,
                'rate': 1e-2,
                'beta': None
            }
        else:
            self.loss_kwargs = loss_kwargs
            
        self.prior = HierarchicalCore(
            latent_dims=latent_dims,
            channels_per_block=channels_per_block,
            input_channels=1,  # Prior takes image only (1 channel)
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name='prior'
        )
        
        self.posterior = HierarchicalCore(
            latent_dims=latent_dims,
            channels_per_block=channels_per_block,
            input_channels=num_classes + 1,  # Posterior takes seg + img (2 + 1 = 3 channels)
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name='posterior'
        )
        
        self.f_comb = StitchingDecoder(
            latent_dims, channels_per_block, num_classes,
            down_channels_per_block, activation_fn, initializers,
            regularizers, convs_per_block, blocks_per_level
        )
        
        if self.loss_kwargs['type'] == 'geco':
            self.moving_average = MovingAverage(
                decay=self.loss_kwargs['decay'], differentiable=True
            )
            self.lagmul = LagrangeMultiplier(rate=self.loss_kwargs['rate'])
            
        self.cache = None
        
    def forward(self, seg, img):
        self._build(seg, img)
        return self._p_sample_z_q
        
    def _build(self, seg, img):
        inputs = (seg, img)
        if self.cache == inputs:
            return
            
        input_cat = torch.cat([seg, img], dim=1)  # Changed from dim=-1 to dim=1
        self._q_sample = self.posterior(input_cat, mean=False)
        self._q_sample_mean = self.posterior(input_cat, mean=True)
        self._p_sample = self.prior(img, mean=False, z_q=None)
        self._p_sample_z_q = self.prior(img, z_q=self._q_sample['used_latents'])
        self._p_sample_z_q_mean = self.prior(img, z_q=self._q_sample_mean['used_latents'])
        self.cache = inputs
        
    def sample(self, img, mean=False, z_q=None):
        prior_out = self.prior(img, mean, z_q)
        return self.f_comb(prior_out['encoder_features'], prior_out['decoder_features'])
        
    def reconstruct(self, seg, img, mean=False):
        self._build(seg, img)
        prior_out = self._p_sample_z_q_mean if mean else self._p_sample_z_q
        return self.f_comb(prior_out['encoder_features'], prior_out['decoder_features'])
        
    def rec_loss(self, seg, img, mask=None, top_k_percentage=None, deterministic=True):
        reconstruction = self.reconstruct(seg, img, mean=False)
        return ce_loss(reconstruction, seg, mask, top_k_percentage, deterministic)
        
    def kl(self, seg, img):
        self._build(seg, img)
        q_dists = self._q_sample['distributions']
        p_dists = self._p_sample_z_q['distributions']
        
        kl = {}
        for level, (q, p) in enumerate(zip(q_dists, p_dists)):
            kl_per_pixel = tfd.kl_divergence(q, p)
            kl_per_instance = kl_per_pixel.sum(dim=[1, 2])
            kl[level] = kl_per_instance.mean()
        return kl
        
    def loss(self, seg, img, mask):
        summaries = {}
        top_k_percentage = self.loss_kwargs['top_k_percentage']
        deterministic = self.loss_kwargs['deterministic_top_k']
        rec_loss = self.rec_loss(seg, img, mask, top_k_percentage, deterministic)
        
        kl_dict = self.kl(seg, img)
        kl_sum = sum(kl_dict.values())
        
        summaries['rec_loss_mean'] = rec_loss['mean']
        summaries['rec_loss_sum'] = rec_loss['sum']
        summaries['kl_sum'] = kl_sum
        for level, kl in kl_dict.items():
            summaries[f'kl_{level}'] = kl
            
        if self.loss_kwargs['type'] == 'elbo':
            loss = rec_loss['sum'] + self.loss_kwargs['beta'] * kl_sum
            summaries['elbo_loss'] = loss
            
        elif self.loss_kwargs['type'] == 'geco':
            ma_rec_loss = self.moving_average(rec_loss['sum'])
            mask_sum_per_instance = rec_loss['mask'].sum(dim=-1)
            num_valid_pixels = mask_sum_per_instance.mean()
            reconstruction_threshold = self.loss_kwargs['kappa'] * num_valid_pixels
            
            rec_constraint = ma_rec_loss - reconstruction_threshold
            lagmul = self.lagmul(rec_constraint)
            loss = lagmul * rec_constraint + kl_sum
            
            summaries['geco_loss'] = loss
            summaries['ma_rec_loss_mean'] = ma_rec_loss / num_valid_pixels
            summaries['num_valid_pixels'] = num_valid_pixels
            summaries['lagmul'] = lagmul
        else:
            raise NotImplementedError(f"Loss type {self.loss_kwargs['type']} not implemented!")
            
        return dict(supervised_loss=loss, summaries=summaries)

if __name__ == '__main__':
    hpu_net = HierarchicalProbUNet()