import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Independent, Normal, kl_divergence

import geco_utils  # your PyTorch version from before
import unet_utils  # assumed to export ResBlock, resize_up, resize_down

###############################################################################
# _HierarchicalCore
###############################################################################

class _HierarchicalCore(nn.Module):
    """
    A U-Net encoder-decoder with a full encoder and a truncated decoder.
    
    The decoder is interleaved with a hierarchical latent space. For each latent
    scale, a 1x1 convolution predicts per-pixel Gaussian parameters.
    
    Note:
      • The network expects inputs in channels-first format: (B, C, H, W).
      • For the “prior” core, inputs are images; for the “posterior”, they are the
        concatenation of segmentation and image.
      • The parameter `input_channels` should be provided (e.g. 1 for grayscale images,
        or seg_channels+img_channels for the posterior).
    """
    def __init__(self, input_channels, latent_dims, channels_per_block,
                 down_channels_per_block=None, activation_fn=F.relu,
                 convs_per_block=3, blocks_per_level=3, upsample_scale=2):
        super(_HierarchicalCore, self).__init__()
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.blocks_per_level = blocks_per_level
        self.convs_per_block = convs_per_block
        self.activation_fn = activation_fn
        self.num_levels = len(channels_per_block)
        self.num_latents = len(latent_dims)
        self.upsample_scale = upsample_scale
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block
        self.down_channels_per_block = down_channels_per_block

        # ----- Encoder: Build a list of residual block groups -----
        # The input to level 0 is of dimension input_channels.
        self.encoder_blocks = nn.ModuleList()
        current_channels = input_channels
        for level in range(self.num_levels):
            # Each level consists of 'blocks_per_level' residual blocks.
            blocks = nn.Sequential(*[
                unet_utils.ResBlock(current_channels, channels_per_block[level],
                                    convs_per_block=convs_per_block,
                                    activation_fn=activation_fn)
                for _ in range(blocks_per_level)
            ])
            self.encoder_blocks.append(blocks)
            current_channels = channels_per_block[level]
            # Downsampling (except after the last level) is done via average pooling.
            # (We use the functional unet_utils.resize_down in forward.)

        # ----- Decoder: For each latent level, build a 1x1 conv and a residual block group.
        self.latent_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        # At the start of decoding, decoder_features = last encoder output;
        # its channels equal channels_per_block[-1].
        for i in range(self.num_latents):
            # Conv that predicts 2xlatent_dim channels from the current decoder features.
            self.latent_convs.append(nn.Conv2d(current_channels, 2 * latent_dims[i], kernel_size=1))
            # After concatenating z with decoder_features, the channel count becomes:
            #   current_channels + latent_dims[i] + (encoder channel from corresponding level)
            # Here we assume the corresponding encoder output comes from level: -(i+2)
            encoder_feat_channels = channels_per_block[-(i + 2)]
            dec_in_channels = current_channels + latent_dims[i] + encoder_feat_channels
            # A residual block group to process the concatenated features.
            self.decoder_blocks.append(nn.Sequential(*[
                unet_utils.ResBlock(dec_in_channels, encoder_feat_channels,
                                    convs_per_block=convs_per_block,
                                    activation_fn=activation_fn)
                for _ in range(blocks_per_level)
            ]))
            # After processing, the new decoder feature channel count becomes that of the encoder feature.
            current_channels = encoder_feat_channels

        self.final_decoder_channels = current_channels  # This will be used later

    def forward(self, inputs, mean=False, z_q=None):
        # ----- Encoder -----
        x = inputs
        encoder_outputs = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            encoder_outputs.append(x)
            if i != self.num_levels - 1:
                x = unet_utils.resize_down(x, scale=self.upsample_scale)
        decoder_features = encoder_outputs[-1]

        # ----- Decoder with latent conditioning -----
        distributions = []
        used_latents = []
        for i in range(self.num_latents):
            conv = self.latent_convs[i]
            mu_logsigma = conv(decoder_features)
            latent_dim = self.latent_dims[i]
            mu = mu_logsigma[:, :latent_dim, :, :]
            logsigma = mu_logsigma[:, latent_dim:, :, :]
            # Create a per-pixel Gaussian (diagonal covariance)
            dist = Independent(Normal(mu, logsigma.exp()), 1)
            distributions.append(dist)
            if z_q is not None:
                z = z_q[i]
            elif (isinstance(mean, bool) and mean) or (isinstance(mean, list) and mean[i]):
                z = mu
            else:
                z = dist.rsample()  # reparameterized sample
            used_latents.append(z)
            # Concatenate z with decoder features along channel dimension.
            decoder_output_lo = torch.cat([z, decoder_features], dim=1)
            decoder_output_hi = unet_utils.resize_up(decoder_output_lo, scale=self.upsample_scale)
            # Get the corresponding encoder feature from the reversed list (index i+1)
            encoder_feature = encoder_outputs[::-1][i + 1]
            decoder_features = torch.cat([decoder_output_hi, encoder_feature], dim=1)
            decoder_features = self.decoder_blocks[i](decoder_features)
        return {'decoder_features': decoder_features,
                'encoder_features': encoder_outputs,
                'distributions': distributions,
                'used_latents': used_latents}

###############################################################################
# _StitchingDecoder
###############################################################################

class _StitchingDecoder(nn.Module):
    """
    Completes the truncated U-Net decoder (from _HierarchicalCore) by
    “stitching” in additional decoder levels to form a symmetric U-Net.
    """
    def __init__(self, latent_dims, channels_per_block, num_classes,
                 down_channels_per_block=None, activation_fn=F.relu,
                 convs_per_block=3, blocks_per_level=3, upsample_scale=2):
        super(_StitchingDecoder, self).__init__()
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.num_classes = num_classes
        self.blocks_per_level = blocks_per_level
        self.convs_per_block = convs_per_block
        self.activation_fn = activation_fn
        self.upsample_scale = upsample_scale
        self.num_latents = len(latent_dims)
        self.num_levels = len(channels_per_block)
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block
        self.down_channels_per_block = down_channels_per_block

        start_level = self.num_latents + 1
        self.stitch_blocks = nn.ModuleList()
        # For each remaining level, we build a residual block group.
        for level in range(start_level, self.num_levels):
            # Here we roughly assume that after upsampling and concatenation the
            # channel count is twice that of the encoder feature at that level.
            in_channels = channels_per_block[::-1][level] * 2  
            self.stitch_blocks.append(nn.Sequential(*[
                unet_utils.ResBlock(in_channels, channels_per_block[::-1][level],
                                    convs_per_block=convs_per_block,
                                    activation_fn=activation_fn)
                for _ in range(blocks_per_level)
            ]))
        # Final 1x1 convolution to produce logits.
        self.logits_conv = nn.Conv2d(channels_per_block[0], num_classes, kernel_size=1)

    def forward(self, encoder_features, decoder_features):
        start_level = self.num_latents + 1
        for i, block in enumerate(self.stitch_blocks):
            decoder_features = unet_utils.resize_up(decoder_features, scale=self.upsample_scale)
            # Select the corresponding encoder feature (from the reversed list)
            encoder_feature = encoder_features[::-1][start_level + i]
            decoder_features = torch.cat([decoder_features, encoder_feature], dim=1)
            decoder_features = block(decoder_features)
        logits = self.logits_conv(decoder_features)
        return logits

###############################################################################
# HierarchicalProbUNet
###############################################################################

class HierarchicalProbUNet(nn.Module):
    """
    Hierarchical Probabilistic U-Net.
    
    This network contains two parallel cores - one for the prior and one for the
    posterior - and a stitching decoder that produces the final segmentation logits.
    
    Note:
      • For the prior core, set `prior_in_channels` to the number of image channels.
      • For the posterior core, set `posterior_in_channels` to the sum of image and
        segmentation channels.
    """
    def __init__(self,
                 latent_dims=(1, 1, 1, 1),
                 channels_per_block=None,
                 num_classes=2,
                 down_channels_per_block=None,
                 activation_fn=F.relu,
                 convs_per_block=3,
                 blocks_per_level=3,
                 loss_kwargs=None,
                 prior_in_channels=1,
                 posterior_in_channels=4):  # e.g. image (1) + segmentation (3)
        super(HierarchicalProbUNet, self).__init__()
        base_channels = 24
        default_channels_per_block = (
            base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels,
            8 * base_channels, 8 * base_channels, 8 * base_channels, 8 * base_channels
        )
        if channels_per_block is None:
            channels_per_block = default_channels_per_block
        if down_channels_per_block is None:
            down_channels_per_block = tuple([int(i / 2) for i in default_channels_per_block])
        if loss_kwargs is None:
            self._loss_kwargs = {
                'type': 'geco',
                'top_k_percentage': 0.02,
                'deterministic_top_k': False,
                'kappa': 0.05,
                'decay': 0.99,
                'rate': 1e-2,
                'beta': None
            }
        else:
            self._loss_kwargs = loss_kwargs
        self.latent_dims = latent_dims

        # Create the two cores.
        self.prior = _HierarchicalCore(input_channels=prior_in_channels,
                                       latent_dims=latent_dims,
                                       channels_per_block=channels_per_block,
                                       down_channels_per_block=down_channels_per_block,
                                       activation_fn=activation_fn,
                                       convs_per_block=convs_per_block,
                                       blocks_per_level=blocks_per_level)
        self.posterior = _HierarchicalCore(input_channels=posterior_in_channels,
                                           latent_dims=latent_dims,
                                           channels_per_block=channels_per_block,
                                           down_channels_per_block=down_channels_per_block,
                                           activation_fn=activation_fn,
                                           convs_per_block=convs_per_block,
                                           blocks_per_level=blocks_per_level)
        self.f_comb = _StitchingDecoder(latent_dims=latent_dims,
                                        channels_per_block=channels_per_block,
                                        num_classes=num_classes,
                                        down_channels_per_block=down_channels_per_block,
                                        activation_fn=activation_fn,
                                        convs_per_block=convs_per_block,
                                        blocks_per_level=blocks_per_level)
        if self._loss_kwargs['type'] == 'geco':
            self._moving_average = geco_utils.MovingAverage(decay=self._loss_kwargs['decay'], differentiable=True)
            self._lagmul = geco_utils.LagrangeMultiplier(rate=self._loss_kwargs['rate'])
        self._cache = None  # simple cache to avoid re-computing

    def _build(self, seg, img):
        inputs = (seg, img)
        if self._cache == inputs:
            return
        else:
            # Concatenate segmentation and image along the channel dimension.
            combined = torch.cat([seg, img], dim=1)
            self._q_sample = self.posterior(combined, mean=False)
            self._q_sample_mean = self.posterior(combined, mean=True)
            self._p_sample = self.prior(img, mean=False, z_q=None)
            self._p_sample_z_q = self.prior(img, z_q=self._q_sample['used_latents'])
            self._p_sample_z_q_mean = self.prior(img, z_q=self._q_sample_mean['used_latents'])
            self._cache = inputs

    def sample(self, img, mean=False, z_q=None):
        prior_out = self.prior(img, mean, z_q)
        encoder_features = prior_out['encoder_features']
        decoder_features = prior_out['decoder_features']
        return self.f_comb(encoder_features, decoder_features)

    def reconstruct(self, seg, img, mean=False):
        self._build(seg, img)
        if mean:
            prior_out = self._p_sample_z_q_mean
        else:
            prior_out = self._p_sample_z_q
        encoder_features = prior_out['encoder_features']
        decoder_features = prior_out['decoder_features']
        return self.f_comb(encoder_features, decoder_features)

    def rec_loss(self, seg, img, mask=None, top_k_percentage=None, deterministic=True):
        reconstruction = self.reconstruct(seg, img, mean=False)
        return geco_utils.ce_loss(reconstruction, seg, mask, top_k_percentage, deterministic)

    def kl(self, seg, img):
        self._build(seg, img)
        posterior_out = self._q_sample
        prior_out = self._p_sample_z_q
        q_dists = posterior_out['distributions']
        p_dists = prior_out['distributions']
        kl_dict = {}
        for level, (q, p) in enumerate(zip(q_dists, p_dists)):
            # Compute per-instance KL divergence (summing over spatial dimensions)
            kl_per_pixel = kl_divergence(q, p)  # shape: (B, H, W)
            kl_per_instance = kl_per_pixel.view(kl_per_pixel.size(0), -1).sum(dim=1)
            kl_dict[level] = kl_per_instance.mean()
        return kl_dict

    def loss(self, seg, img, mask):
        summaries = {}
        top_k_percentage = self._loss_kwargs['top_k_percentage']
        deterministic = self._loss_kwargs['deterministic_top_k']
        rec_loss = self.rec_loss(seg, img, mask, top_k_percentage, deterministic)
        kl_dict = self.kl(seg, img)
        kl_sum = sum(kl_dict.values())
        # For compatibility, we assume rec_loss is a dict with keys 'mean' and 'sum'
        summaries['rec_loss_mean'] = rec_loss['mean'] if isinstance(rec_loss, dict) and 'mean' in rec_loss else rec_loss
        summaries['rec_loss_sum'] = rec_loss['sum'] if isinstance(rec_loss, dict) and 'sum' in rec_loss else rec_loss
        summaries['kl_sum'] = kl_sum
        for level, kl_val in kl_dict.items():
            summaries[f'kl_{level}'] = kl_val
        if self._loss_kwargs['type'] == 'elbo':
            loss_val = rec_loss['sum'] + self._loss_kwargs['beta'] * kl_sum
            summaries['elbo_loss'] = loss_val
        elif self._loss_kwargs['type'] == 'geco':
            ma_rec_loss = self._moving_average(rec_loss['sum'])
            mask_sum_per_instance = mask.view(mask.size(0), -1).sum(dim=1)
            num_valid_pixels = mask_sum_per_instance.mean()
            reconstruction_threshold = self._loss_kwargs['kappa'] * num_valid_pixels
            rec_constraint = ma_rec_loss - reconstruction_threshold
            lagmul = self._lagmul(rec_constraint)
            loss_val = lagmul * rec_constraint + kl_sum
            summaries['geco_loss'] = loss_val
            summaries['ma_rec_loss_mean'] = ma_rec_loss / num_valid_pixels
            summaries['num_valid_pixels'] = num_valid_pixels
            summaries['lagmul'] = lagmul
        else:
            raise NotImplementedError(f"Loss type {self._loss_kwargs['type']} not implemented!")
        return {'supervised_loss': loss_val, 'summaries': summaries}

if __name__ == '__main__':
    model = HierarchicalProbUNet()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Independent, Normal, kl_divergence

import geco_utils  # your PyTorch version from before
import unet_utils  # assumed to export ResBlock, resize_up, resize_down

###############################################################################
# _HierarchicalCore
###############################################################################

class _HierarchicalCore(nn.Module):
    """
    A U-Net encoder-decoder with a full encoder and a truncated decoder.
    
    The decoder is interleaved with a hierarchical latent space. For each latent
    scale, a 1x1 convolution predicts per-pixel Gaussian parameters.
    
    Note:
      • The network expects inputs in channels-first format: (B, C, H, W).
      • For the “prior” core, inputs are images; for the “posterior”, they are the
        concatenation of segmentation and image.
      • The parameter `input_channels` should be provided (e.g. 1 for grayscale images,
        or seg_channels+img_channels for the posterior).
    """
    def __init__(self, input_channels, latent_dims, channels_per_block,
                 down_channels_per_block=None, activation_fn=F.relu,
                 convs_per_block=3, blocks_per_level=3, upsample_scale=2):
        super(_HierarchicalCore, self).__init__()
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.blocks_per_level = blocks_per_level
        self.convs_per_block = convs_per_block
        self.activation_fn = activation_fn
        self.num_levels = len(channels_per_block)
        self.num_latents = len(latent_dims)
        self.upsample_scale = upsample_scale
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block
        self.down_channels_per_block = down_channels_per_block

        # ----- Encoder: Build a list of residual block groups -----
        # The input to level 0 is of dimension input_channels.
        self.encoder_blocks = nn.ModuleList()
        current_channels = input_channels
        for level in range(self.num_levels):
            # Each level consists of 'blocks_per_level' residual blocks.
            blocks = nn.Sequential(*[
                unet_utils.ResBlock(current_channels, channels_per_block[level],
                                    convs_per_block=convs_per_block,
                                    activation_fn=activation_fn)
                for _ in range(blocks_per_level)
            ])
            self.encoder_blocks.append(blocks)
            current_channels = channels_per_block[level]
            # Downsampling (except after the last level) is done via average pooling.
            # (We use the functional unet_utils.resize_down in forward.)

        # ----- Decoder: For each latent level, build a 1x1 conv and a residual block group.
        self.latent_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        # At the start of decoding, decoder_features = last encoder output;
        # its channels equal channels_per_block[-1].
        for i in range(self.num_latents):
            # Conv that predicts 2xlatent_dim channels from the current decoder features.
            self.latent_convs.append(nn.Conv2d(current_channels, 2 * latent_dims[i], kernel_size=1))
            # After concatenating z with decoder_features, the channel count becomes:
            #   current_channels + latent_dims[i] + (encoder channel from corresponding level)
            # Here we assume the corresponding encoder output comes from level: -(i+2)
            encoder_feat_channels = channels_per_block[-(i + 2)]
            dec_in_channels = current_channels + latent_dims[i] + encoder_feat_channels
            # A residual block group to process the concatenated features.
            self.decoder_blocks.append(nn.Sequential(*[
                unet_utils.ResBlock(dec_in_channels, encoder_feat_channels,
                                    convs_per_block=convs_per_block,
                                    activation_fn=activation_fn)
                for _ in range(blocks_per_level)
            ]))
            # After processing, the new decoder feature channel count becomes that of the encoder feature.
            current_channels = encoder_feat_channels

        self.final_decoder_channels = current_channels  # This will be used later

    def forward(self, inputs, mean=False, z_q=None):
        # ----- Encoder -----
        x = inputs
        encoder_outputs = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            encoder_outputs.append(x)
            if i != self.num_levels - 1:
                x = unet_utils.resize_down(x, scale=self.upsample_scale)
        decoder_features = encoder_outputs[-1]

        # ----- Decoder with latent conditioning -----
        distributions = []
        used_latents = []
        for i in range(self.num_latents):
            conv = self.latent_convs[i]
            mu_logsigma = conv(decoder_features)
            latent_dim = self.latent_dims[i]
            mu = mu_logsigma[:, :latent_dim, :, :]
            logsigma = mu_logsigma[:, latent_dim:, :, :]
            # Create a per-pixel Gaussian (diagonal covariance)
            dist = Independent(Normal(mu, logsigma.exp()), 1)
            distributions.append(dist)
            if z_q is not None:
                z = z_q[i]
            elif (isinstance(mean, bool) and mean) or (isinstance(mean, list) and mean[i]):
                z = mu
            else:
                z = dist.rsample()  # reparameterized sample
            used_latents.append(z)
            # Concatenate z with decoder features along channel dimension.
            decoder_output_lo = torch.cat([z, decoder_features], dim=1)
            decoder_output_hi = unet_utils.resize_up(decoder_output_lo, scale=self.upsample_scale)
            # Get the corresponding encoder feature from the reversed list (index i+1)
            encoder_feature = encoder_outputs[::-1][i + 1]
            decoder_features = torch.cat([decoder_output_hi, encoder_feature], dim=1)
            decoder_features = self.decoder_blocks[i](decoder_features)
        return {'decoder_features': decoder_features,
                'encoder_features': encoder_outputs,
                'distributions': distributions,
                'used_latents': used_latents}

###############################################################################
# _StitchingDecoder
###############################################################################

class _StitchingDecoder(nn.Module):
    """
    Completes the truncated U-Net decoder (from _HierarchicalCore) by
    “stitching” in additional decoder levels to form a symmetric U-Net.
    """
    def __init__(self, latent_dims, channels_per_block, num_classes,
                 down_channels_per_block=None, activation_fn=F.relu,
                 convs_per_block=3, blocks_per_level=3, upsample_scale=2):
        super(_StitchingDecoder, self).__init__()
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.num_classes = num_classes
        self.blocks_per_level = blocks_per_level
        self.convs_per_block = convs_per_block
        self.activation_fn = activation_fn
        self.upsample_scale = upsample_scale
        self.num_latents = len(latent_dims)
        self.num_levels = len(channels_per_block)
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block
        self.down_channels_per_block = down_channels_per_block

        start_level = self.num_latents + 1
        self.stitch_blocks = nn.ModuleList()
        # For each remaining level, we build a residual block group.
        for level in range(start_level, self.num_levels):
            # Here we roughly assume that after upsampling and concatenation the
            # channel count is twice that of the encoder feature at that level.
            in_channels = channels_per_block[::-1][level] * 2  
            self.stitch_blocks.append(nn.Sequential(*[
                unet_utils.ResBlock(in_channels, channels_per_block[::-1][level],
                                    convs_per_block=convs_per_block,
                                    activation_fn=activation_fn)
                for _ in range(blocks_per_level)
            ]))
        # Final 1x1 convolution to produce logits.
        self.logits_conv = nn.Conv2d(channels_per_block[0], num_classes, kernel_size=1)

    def forward(self, encoder_features, decoder_features):
        start_level = self.num_latents + 1
        for i, block in enumerate(self.stitch_blocks):
            decoder_features = unet_utils.resize_up(decoder_features, scale=self.upsample_scale)
            # Select the corresponding encoder feature (from the reversed list)
            encoder_feature = encoder_features[::-1][start_level + i]
            decoder_features = torch.cat([decoder_features, encoder_feature], dim=1)
            decoder_features = block(decoder_features)
        logits = self.logits_conv(decoder_features)
        return logits

###############################################################################
# HierarchicalProbUNet
###############################################################################

class HierarchicalProbUNet(nn.Module):
    """
    Hierarchical Probabilistic U-Net.
    
    This network contains two parallel cores - one for the prior and one for the
    posterior - and a stitching decoder that produces the final segmentation logits.
    
    Note:
      • For the prior core, set `prior_in_channels` to the number of image channels.
      • For the posterior core, set `posterior_in_channels` to the sum of image and
        segmentation channels.
    """
    def __init__(self,
                 latent_dims=(1, 1, 1, 1),
                 channels_per_block=None,
                 num_classes=2,
                 down_channels_per_block=None,
                 activation_fn=F.relu,
                 convs_per_block=3,
                 blocks_per_level=3,
                 loss_kwargs=None,
                 prior_in_channels=1,
                 posterior_in_channels=4):  # e.g. image (1) + segmentation (3)
        super(HierarchicalProbUNet, self).__init__()
        base_channels = 24
        default_channels_per_block = (
            base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels,
            8 * base_channels, 8 * base_channels, 8 * base_channels, 8 * base_channels
        )
        if channels_per_block is None:
            channels_per_block = default_channels_per_block
        if down_channels_per_block is None:
            down_channels_per_block = tuple([int(i / 2) for i in default_channels_per_block])
        if loss_kwargs is None:
            self._loss_kwargs = {
                'type': 'geco',
                'top_k_percentage': 0.02,
                'deterministic_top_k': False,
                'kappa': 0.05,
                'decay': 0.99,
                'rate': 1e-2,
                'beta': None
            }
        else:
            self._loss_kwargs = loss_kwargs
        self.latent_dims = latent_dims

        # Create the two cores.
        self.prior = _HierarchicalCore(input_channels=prior_in_channels,
                                       latent_dims=latent_dims,
                                       channels_per_block=channels_per_block,
                                       down_channels_per_block=down_channels_per_block,
                                       activation_fn=activation_fn,
                                       convs_per_block=convs_per_block,
                                       blocks_per_level=blocks_per_level)
        self.posterior = _HierarchicalCore(input_channels=posterior_in_channels,
                                           latent_dims=latent_dims,
                                           channels_per_block=channels_per_block,
                                           down_channels_per_block=down_channels_per_block,
                                           activation_fn=activation_fn,
                                           convs_per_block=convs_per_block,
                                           blocks_per_level=blocks_per_level)
        self.f_comb = _StitchingDecoder(latent_dims=latent_dims,
                                        channels_per_block=channels_per_block,
                                        num_classes=num_classes,
                                        down_channels_per_block=down_channels_per_block,
                                        activation_fn=activation_fn,
                                        convs_per_block=convs_per_block,
                                        blocks_per_level=blocks_per_level)
        if self._loss_kwargs['type'] == 'geco':
            self._moving_average = geco_utils.MovingAverage(decay=self._loss_kwargs['decay'], differentiable=True)
            self._lagmul = geco_utils.LagrangeMultiplier(rate=self._loss_kwargs['rate'])
        self._cache = None  # simple cache to avoid re-computing

    def _build(self, seg, img):
        inputs = (seg, img)
        if self._cache == inputs:
            return
        else:
            # Concatenate segmentation and image along the channel dimension.
            combined = torch.cat([seg, img], dim=1)
            self._q_sample = self.posterior(combined, mean=False)
            self._q_sample_mean = self.posterior(combined, mean=True)
            self._p_sample = self.prior(img, mean=False, z_q=None)
            self._p_sample_z_q = self.prior(img, z_q=self._q_sample['used_latents'])
            self._p_sample_z_q_mean = self.prior(img, z_q=self._q_sample_mean['used_latents'])
            self._cache = inputs

    def sample(self, img, mean=False, z_q=None):
        prior_out = self.prior(img, mean, z_q)
        encoder_features = prior_out['encoder_features']
        decoder_features = prior_out['decoder_features']
        return self.f_comb(encoder_features, decoder_features)

    def reconstruct(self, seg, img, mean=False):
        self._build(seg, img)
        if mean:
            prior_out = self._p_sample_z_q_mean
        else:
            prior_out = self._p_sample_z_q
        encoder_features = prior_out['encoder_features']
        decoder_features = prior_out['decoder_features']
        return self.f_comb(encoder_features, decoder_features)

    def rec_loss(self, seg, img, mask=None, top_k_percentage=None, deterministic=True):
        reconstruction = self.reconstruct(seg, img, mean=False)
        return geco_utils.ce_loss(reconstruction, seg, mask, top_k_percentage, deterministic)

    def kl(self, seg, img):
        self._build(seg, img)
        posterior_out = self._q_sample
        prior_out = self._p_sample_z_q
        q_dists = posterior_out['distributions']
        p_dists = prior_out['distributions']
        kl_dict = {}
        for level, (q, p) in enumerate(zip(q_dists, p_dists)):
            # Compute per-instance KL divergence (summing over spatial dimensions)
            kl_per_pixel = kl_divergence(q, p)  # shape: (B, H, W)
            kl_per_instance = kl_per_pixel.view(kl_per_pixel.size(0), -1).sum(dim=1)
            kl_dict[level] = kl_per_instance.mean()
        return kl_dict

    def loss(self, seg, img, mask):
        summaries = {}
        top_k_percentage = self._loss_kwargs['top_k_percentage']
        deterministic = self._loss_kwargs['deterministic_top_k']
        rec_loss = self.rec_loss(seg, img, mask, top_k_percentage, deterministic)
        kl_dict = self.kl(seg, img)
        kl_sum = sum(kl_dict.values())
        # For compatibility, we assume rec_loss is a dict with keys 'mean' and 'sum'
        summaries['rec_loss_mean'] = rec_loss['mean'] if isinstance(rec_loss, dict) and 'mean' in rec_loss else rec_loss
        summaries['rec_loss_sum'] = rec_loss['sum'] if isinstance(rec_loss, dict) and 'sum' in rec_loss else rec_loss
        summaries['kl_sum'] = kl_sum
        for level, kl_val in kl_dict.items():
            summaries[f'kl_{level}'] = kl_val
        if self._loss_kwargs['type'] == 'elbo':
            loss_val = rec_loss['sum'] + self._loss_kwargs['beta'] * kl_sum
            summaries['elbo_loss'] = loss_val
        elif self._loss_kwargs['type'] == 'geco':
            ma_rec_loss = self._moving_average(rec_loss['sum'])
            mask_sum_per_instance = mask.view(mask.size(0), -1).sum(dim=1)
            num_valid_pixels = mask_sum_per_instance.mean()
            reconstruction_threshold = self._loss_kwargs['kappa'] * num_valid_pixels
            rec_constraint = ma_rec_loss - reconstruction_threshold
            lagmul = self._lagmul(rec_constraint)
            loss_val = lagmul * rec_constraint + kl_sum
            summaries['geco_loss'] = loss_val
            summaries['ma_rec_loss_mean'] = ma_rec_loss / num_valid_pixels
            summaries['num_valid_pixels'] = num_valid_pixels
            summaries['lagmul'] = lagmul
        else:
            raise NotImplementedError(f"Loss type {self._loss_kwargs['type']} not implemented!")
        return {'supervised_loss': loss_val, 'summaries': summaries}

if __name__ == '__main__':
    model = HierarchicalProbUNet()
