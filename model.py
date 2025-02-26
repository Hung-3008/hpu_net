import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

def resize_down(x, scale=2):
    return F.avg_pool2d(x, kernel_size=scale, stride=scale)

def resize_up(x, scale=2):
    return F.interpolate(x, scale_factor=scale, mode='nearest')

# ------------------------------
# Pre-allocated ResidualBlock Module
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, intermediate_channels=None, convs_per_block=3, activation_fn=F.relu):
        super().__init__()
        if intermediate_channels is None:
            intermediate_channels = out_channels
        layers = []
        layers.append(nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for i in range(convs_per_block - 1):
            layers.append(nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1))
            if i < convs_per_block - 2:
                layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.out_conv = nn.Conv2d(intermediate_channels, out_channels, kernel_size=1) if intermediate_channels != out_channels else nn.Identity()
        self.activation_fn = activation_fn

    def forward(self, x):
        skip = self.skip_conv(x)
        out = self.activation_fn(x)
        out = self.conv(out)
        out = self.out_conv(out)
        return out + skip

# ------------------------------
# HierarchicalCore Module with Pre-allocated Layers
# ------------------------------
class HierarchicalCore(nn.Module):
    def __init__(self, latent_dims, channels_per_block, down_channels_per_block=None,
                 activation_fn=F.relu, convs_per_block=3, blocks_per_level=3, in_channels=1):
        """
        channels_per_block: list of length L (e.g., [5,7,9,11,13])
        latent_dims: list of length M (e.g., [3,2,1])
        """
        super().__init__()
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.down_channels_per_block = down_channels_per_block if down_channels_per_block is not None else channels_per_block
        self.activation_fn = activation_fn
        self.convs_per_block = convs_per_block
        self.blocks_per_level = blocks_per_level
        self.num_levels = len(channels_per_block)
        self.num_latent_levels = len(latent_dims)
        
        # Encoder
        self.input_conv = nn.Conv2d(in_channels, channels_per_block[0], kernel_size=3, padding=1)
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i in range(self.num_levels):
            block_layers = []
            # For level 0, input is already mapped to channels_per_block[0]
            if i > 0:
                # Transition convolution: map from previous level's channels to current
                block_layers.append(nn.Conv2d(channels_per_block[i-1], channels_per_block[i], kernel_size=3, padding=1))
            for _ in range(blocks_per_level):
                block_layers.append(ResidualBlock(channels_per_block[i], channels_per_block[i],
                                                  intermediate_channels=self.down_channels_per_block[i],
                                                  convs_per_block=convs_per_block, activation_fn=activation_fn))
            self.encoder_blocks.append(nn.Sequential(*block_layers))
            if i < self.num_levels - 1:
                self.downsamplers.append(nn.AvgPool2d(2))
        
        # Decoder
        self.latent_conv = nn.ModuleList() 
        self.decoder_refinement = nn.ModuleList()
        current_ch = self.channels_per_block[-1]  # from deepest encoder (level L-1)
        for i in range(self.num_latent_levels):
            ld = latent_dims[i]
            self.latent_conv.append(nn.Conv2d(current_ch, 2 * ld, kernel_size=1, padding=0))
            injected_ch = current_ch + ld
            skip_ch = self.channels_per_block[self.num_levels - 1 - (i+1)]
            total_ch = injected_ch + skip_ch
            blocks = []
            blocks.append(ResidualBlock(
                total_ch,
                skip_ch,
                intermediate_channels=self.down_channels_per_block[::-1][i+1],
                convs_per_block=convs_per_block,
                activation_fn=activation_fn
            ))
            for _ in range(blocks_per_level - 1):
                blocks.append(ResidualBlock(
                    skip_ch,
                    skip_ch,
                    intermediate_channels=self.down_channels_per_block[::-1][i+1],
                    convs_per_block=convs_per_block,
                    activation_fn=activation_fn
                ))
            self.decoder_refinement.append(nn.Sequential(*blocks))
            current_ch = skip_ch

    def forward(self, inputs, mean=False, z_q=None):
        # Encoder forward
        x = self.input_conv(inputs)
        encoder_outputs = []
        for i in range(self.num_levels):
            x = self.encoder_blocks[i](x)
            encoder_outputs.append(x)
            if i < self.num_levels - 1:
                x = self.downsamplers[i](x)
        # Decoder forward: start with deepest encoder output.
        decoder_features = encoder_outputs[-1]
        distributions = []
        used_latents = []
        for i in range(self.num_latent_levels):
            conv1x1 = self.latent_conv[i]
            mu_logsigma = conv1x1(decoder_features)
            ld = self.latent_dims[i]
            mu = mu_logsigma[:, :ld, :, :]
            logsigma = mu_logsigma[:, ld:, :, :]
            dist = Independent(Normal(mu, torch.exp(logsigma)), 1)
            distributions.append(dist)
            if z_q is not None:
                z = z_q[i]
            elif isinstance(mean, bool) and mean:
                z = mu
            elif isinstance(mean, list) and mean[i]:
                z = mu
            else:
                z = dist.rsample()
            used_latents.append(z)
            # Concatenate latent and current features.
            decoder_features = torch.cat([z, decoder_features], dim=1)
            decoder_features = resize_up(decoder_features, scale=2)
            # Get corresponding skip connection (from reversed encoder outputs)
            skip_feat = encoder_outputs[::-1][i+1]
            # Ensure spatial sizes match:
            if decoder_features.shape[2:] != skip_feat.shape[2:]:
                skip_feat = F.interpolate(skip_feat, size=decoder_features.shape[2:], mode='nearest')
            decoder_features = torch.cat([decoder_features, skip_feat], dim=1)
            decoder_features = self.decoder_refinement[i](decoder_features)
        return {'decoder_features': decoder_features,
                'encoder_features': encoder_outputs,
                'distributions': distributions,
                'used_latents': used_latents}


# ------------------------------
# StitchingDecoder Module with Pre-allocated Layers - FIXED VERSION
# ------------------------------
class StitchingDecoder(nn.Module):
    """
    A module that completes the truncated U-Net decoder.
    
    Using the output of the HierarchicalCore (i.e. the encoder outputs and the truncated decoder features),
    this module upsamples and concatenates skip connections to form a full symmetric U-Net, outputting
    segmentation logits.
    """
    def __init__(self, latent_dims, channels_per_block, num_classes,
                 down_channels_per_block=None, activation_fn=F.relu,
                 convs_per_block=3, blocks_per_level=3):
        super().__init__()
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.num_classes = num_classes
        self.activation_fn = activation_fn
        self.convs_per_block = convs_per_block
        self.blocks_per_level = blocks_per_level
        self.down_channels_per_block = down_channels_per_block if down_channels_per_block is not None else channels_per_block
        
        self.num_levels = len(channels_per_block)
        self.num_latents = len(latent_dims)
        self.start_level = self.num_latents + 1
        
        # Pre-allocate transition convs and refinement blocks for remaining levels
        self.decoder_blocks = nn.ModuleList()
        self.transition_convs = nn.ModuleList()
        rev_channels = list(reversed(channels_per_block))
        
        for level in range(self.start_level, self.num_levels):
            # Current features channel count
            curr_ch = rev_channels[level - 1]  # Previous level's channels
            # Skip connection channel count
            skip_ch = rev_channels[level]
            # After concatenation, total channels
            total_ch = curr_ch + skip_ch
            
            # Transition conv to reduce channels after concatenation
            self.transition_convs.append(nn.Conv2d(total_ch, skip_ch, kernel_size=3, padding=1))
            
            # Refinement blocks
            blocks = []
            for _ in range(blocks_per_level):
                blocks.append(ResidualBlock(
                    skip_ch, skip_ch,
                    intermediate_channels=self.down_channels_per_block[::-1][level],
                    convs_per_block=convs_per_block,
                    activation_fn=activation_fn
                ))
            self.decoder_blocks.append(nn.Sequential(*blocks))
            
        self.final_conv = nn.Conv2d(rev_channels[-1], num_classes, kernel_size=1, padding=0)

    def forward(self, encoder_features, decoder_features):
        # Reverse the encoder outputs (so that index 0 corresponds to the deepest encoder output)
        encoder_features_reversed = encoder_features[::-1]
        
        # Loop over levels from start_level to num_levels-1
        for level in range(self.start_level, self.num_levels):
            # Upscale decoder features
            decoder_features = resize_up(decoder_features, scale=2)
            
            # Get skip connection from encoder
            skip_connection = encoder_features_reversed[level]
            
            # Ensure spatial dimensions match
            if decoder_features.shape[2:] != skip_connection.shape[2:]:
                skip_connection = F.interpolate(skip_connection, 
                                               size=decoder_features.shape[2:], 
                                               mode='nearest')
            
            # Concatenate upscaled features with skip connection
            decoder_features = torch.cat([decoder_features, skip_connection], dim=1)
            
            # Apply transition conv to reduce channels
            decoder_features = self.transition_convs[level - self.start_level](decoder_features)
            
            # Apply refinement blocks
            decoder_features = self.decoder_blocks[level - self.start_level](decoder_features)
            
        # Final 1x1 conv to get logits
        logits = self.final_conv(decoder_features)
        return logits


# ------------------------------
# GECO Utilities
# ------------------------------
class MovingAverage(nn.Module):
    def __init__(self, decay=0.99, differentiable=True):
        super().__init__()
        self.decay = decay
        self.value = None

    def forward(self, x):
        if self.value is None:
            self.value = x.detach()
        else:
            self.value = self.decay * self.value + (1 - self.decay) * x.detach()
        return self.value

class LagrangeMultiplier(nn.Module):
    def __init__(self, rate=1e-2):
        super().__init__()
        self.rate = rate
        self.multiplier = nn.Parameter(torch.tensor(1.0))

    def forward(self, constraint):
        return self.multiplier

# ------------------------------
# HierarchicalProbUNet Module using Pre-allocated Submodules
# ------------------------------
class HierarchicalProbUNet(nn.Module):
    def __init__(self,
                 latent_dims=(1, 1, 1, 1),
                 channels_per_block=None,
                 num_classes=4,  # For ACDC: background, LV, RV, Myo
                 down_channels_per_block=None,
                 activation_fn=F.relu,
                 convs_per_block=3,
                 blocks_per_level=3,
                 loss_kwargs=None,
                 name='HPUNet',
                 in_channels=1):
        super().__init__()
        base_channels = 24
        default_channels_per_block = (
            base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels,
            8 * base_channels, 8 * base_channels, 8 * base_channels,
            8 * base_channels
        )
        if channels_per_block is None:
            channels_per_block = default_channels_per_block
        if down_channels_per_block is None:
            down_channels_per_block = tuple(i // 2 for i in default_channels_per_block)
        
        if loss_kwargs is None:
            self._loss_kwargs = {
                'type': 'geco',
                'top_k_percentage': 0.02,
                'deterministic_top_k': False,
                'kappa': 0.05,
                'decay': 0.99,
                'rate': 1e-2,
                'beta': 1.0
            }
        else:
            self._loss_kwargs = loss_kwargs

        # Prior network: input is image only.
        self.prior = HierarchicalCore(
            latent_dims=latent_dims,
            channels_per_block=channels_per_block,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            in_channels=in_channels
        )
        # Posterior network: input is image concatenated with segmentation (in one-hot, so channels = in_channels+num_classes)
        self.posterior = HierarchicalCore(
            latent_dims=latent_dims,
            channels_per_block=channels_per_block,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            in_channels=in_channels + num_classes
        )
        self.f_comb = StitchingDecoder(
            latent_dims=latent_dims,
            channels_per_block=channels_per_block,
            num_classes=num_classes,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level
        )
        if self._loss_kwargs['type'] == 'geco':
            self._moving_average = MovingAverage(decay=self._loss_kwargs['decay'], differentiable=True)
            self._lagmul = LagrangeMultiplier(rate=self._loss_kwargs['rate'])
        self._cache = None

    def _build(self, seg, img):
        # If seg is (B, H, W), convert to one-hot: (B, num_classes, H, W)
        if seg.dim() == 3:
            seg = F.one_hot(seg, num_classes=self.f_comb.num_classes).permute(0, 3, 1, 2).float()
        x = torch.cat([seg, img], dim=1)
        self._q_sample = self.posterior(x, mean=False)
        self._q_sample_mean = self.posterior(x, mean=True)
        self._p_sample = self.prior(img, mean=False, z_q=None)
        self._p_sample_z_q = self.prior(img, mean=False, z_q=self._q_sample['used_latents'])
        self._p_sample_z_q_mean = self.prior(img, mean=False, z_q=self._q_sample_mean['used_latents'])

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
        reconstruction = self.reconstruct(seg, img, mean=deterministic)  # (B, num_classes, H, W)
        if seg.dim() == 4:
            seg_target = seg.argmax(dim=1)
        else:
            seg_target = seg
        loss_val = F.cross_entropy(reconstruction, seg_target, reduction='mean')
        return loss_val

    def kl(self, seg, img):
        self._build(seg, img)
        q_dists = self._q_sample['distributions']
        p_dists = self._p_sample_z_q['distributions']
        kl_dict = {}
        for level, (q, p) in enumerate(zip(q_dists, p_dists)):
            kl_per_pixel = torch.distributions.kl_divergence(q, p)
            kl_per_instance = kl_per_pixel.view(kl_per_pixel.shape[0], -1).sum(dim=1)
            kl_dict[level] = kl_per_instance.mean()
        return kl_dict

    def loss(self, seg, img, mask=None):
        summaries = {}
        rec_loss_val = self.rec_loss(seg, img, mask)
        kl_dict = self.kl(seg, img)
        kl_sum = sum(kl_dict.values())
        summaries['rec_loss'] = rec_loss_val.item()
        summaries['kl_sum'] = kl_sum.item()
        for level, kl_val in kl_dict.items():
            summaries[f'kl_{level}'] = kl_val.item()
        if self._loss_kwargs['type'] == 'elbo':
            beta = self._loss_kwargs.get('beta', 1.0)
            loss_val = rec_loss_val + beta * kl_sum
            summaries['elbo_loss'] = loss_val.item()
        elif self._loss_kwargs['type'] == 'geco':
            ma_rec_loss = self._moving_average(rec_loss_val)
            reconstruction_threshold = self._loss_kwargs['kappa']
            rec_constraint = ma_rec_loss - reconstruction_threshold
            lagmul = self._lagmul(rec_constraint)
            loss_val = lagmul * rec_constraint + kl_sum
            summaries['geco_loss'] = loss_val.item()
            summaries['ma_rec_loss'] = ma_rec_loss.item()
            summaries['lagmul'] = lagmul.item()
        else:
            raise NotImplementedError(f"Loss type {self._loss_kwargs['type']} not implemented!")
        return {'supervised_loss': loss_val, 'summaries': summaries}