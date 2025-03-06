import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from geco import *


class ResBlock(nn.Module):
    def __init__(self, input_channels, n_channels, n_down_channels, convs_per_block, activation_fn):
        super().__init__()
        layers = []
        in_channels = input_channels
        for i in range(convs_per_block):
            out_channels = n_down_channels if i < convs_per_block - 1 else n_channels
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            layers.append(conv)
            if i < convs_per_block - 1:
                layers.append(activation_fn())
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*layers)
        self.shortcut = nn.Conv2d(input_channels, n_channels, kernel_size=1) if input_channels != n_channels else nn.Identity()
        self.activation = activation_fn()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_layers(x)
        out += residual
        out = self.activation(out)
        return out


def resize_down(x, scale):
    return F.interpolate(x, scale_factor=1/scale, mode='bilinear', align_corners=False)

def resize_up(x, scale):
    return F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)


class _HierarchicalCore(nn.Module):
    def __init__(self, input_channels, latent_dims, channels_per_block,      down_channels_per_block=None, activation_fn=nn.ReLU, initializers=None, convs_per_block=3, blocks_per_level=3, name='prior'):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.down_channels_per_block = down_channels_per_block if down_channels_per_block is not None else channels_per_block
        self.activation_fn = activation_fn
        self.initializers = initializers
        self.convs_per_block = convs_per_block
        self.blocks_per_level = blocks_per_level
        self.num_levels = len(channels_per_block)
        self.num_latent_levels = len(latent_dims)
        self.name = name

        # Encoder layers
        self.encoder_levels = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        current_channels = input_channels
        for level in range(self.num_levels):
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                block = ResBlock(
                    input_channels=current_channels,
                    n_channels=self.channels_per_block[level],
                    n_down_channels=self.down_channels_per_block[level],
                    convs_per_block=convs_per_block,
                    activation_fn=activation_fn
                )
                blocks.append(block)
                current_channels = self.channels_per_block[level]
            self.encoder_levels.append(blocks)
            if level != self.num_levels - 1:
                downsample = nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1)
                self.downsample_layers.append(downsample)
            else:
                self.downsample_layers.append(None)

        # Decoder layers
        self.mu_logsigma_convs = nn.ModuleList()
        self.decoder_upsample_layers = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()
        decoder_input_channels = self.channels_per_block[-1]
        for level in range(self.num_latent_levels):
            latent_dim = self.latent_dims[level]
            mu_logsigma_conv = nn.Conv2d(decoder_input_channels, 2 * latent_dim, kernel_size=1)
            self.mu_logsigma_convs.append(mu_logsigma_conv)
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(decoder_input_channels + latent_dim, decoder_input_channels + latent_dim, kernel_size=3, padding=1)
            )
            self.decoder_upsample_layers.append(upsample)
            concat_channels = (decoder_input_channels + latent_dim) + self.channels_per_block[self.num_levels - level - 2]
            blocks = nn.ModuleList()
            for i in range(blocks_per_level):
                in_channels = concat_channels if i == 0 else self.channels_per_block[self.num_levels - level - 2]
                block = ResBlock(
                    input_channels=in_channels,
                    n_channels=self.channels_per_block[self.num_levels - level - 2],
                    n_down_channels=self.down_channels_per_block[::-1][level + 1],
                    convs_per_block=convs_per_block,
                    activation_fn=activation_fn
                )
                blocks.append(block)
            self.decoder_levels.append(blocks)
            decoder_input_channels = self.channels_per_block[self.num_levels - level - 2]
            #print(f'Level {level} decoder_input_channels: {decoder_input_channels}')

    def forward(self, inputs, mean=False, z_q=None):
        encoder_features = inputs
        encoder_outputs = []
        mean = [mean] * self.num_latent_levels if isinstance(mean, bool) else mean
        distributions = []
        used_latents = []

        # Encoder forward
        for level in range(self.num_levels):

            for block in self.encoder_levels[level]:
                encoder_features = block(encoder_features)
                #print level and shape of encoder_features
                #print(f'Level {level} encoder_features shape: {encoder_features.shape}')
            encoder_outputs.append(encoder_features)
            if level != self.num_levels - 1:
                encoder_features = self.downsample_layers[level](encoder_features)

        # Decoder forward
        decoder_features = encoder_outputs[-1]
        for level in range(self.num_latent_levels):

            mu_logsigma = self.mu_logsigma_convs[level](decoder_features)
            latent_dim = self.latent_dims[level]
            mu, logsigma = torch.split(mu_logsigma, latent_dim, dim=1)
            dist = torch.distributions.Independent(
                torch.distributions.Normal(loc=mu, scale=torch.exp(logsigma)),
                1
            )
            distributions.append(dist)
            if z_q is not None:
                z = z_q[level]
            elif mean[level]:
                z = mu
            else:
                z = dist.rsample()
            used_latents.append(z)
            decoder_output_lo = torch.cat([z, decoder_features], dim=1)
            decoder_output_hi = self.decoder_upsample_layers[level](decoder_output_lo)
            encoder_feature = encoder_outputs[::-1][level + 1]
            
            decoder_features = torch.cat([decoder_output_hi, encoder_feature], dim=1)
            for block in self.decoder_levels[level]:
                #print(f'Level {level} decoder_features shape: {decoder_features.shape}')
                decoder_features = block(decoder_features)

        return {
            'decoder_features': decoder_features,
            'encoder_features': encoder_outputs,
            'distributions': distributions,
            'used_latents': used_latents
        }


class _StitchingDecoder(nn.Module):
    """
    A PyTorch module that completes the truncated U-Net decoder.

    Using the output of the HierarchicalCore, this module fills in the missing
    decoder levels such that together they form a symmetric U-Net.
    """
    def __init__(self, latent_dims, channels_per_block, num_classes,
                 down_channels_per_block=None, activation_fn=F.relu,
                 convs_per_block=3, blocks_per_level=3, name='f_comb'):
        """
        Initializes the _StitchingDecoder.

        Args:
            latent_dims (list of int): Dimensions of the latents at each scale. The length
                indicates the number of U-Net decoder scales with latents.
            channels_per_block (list of int): Number of output channels for each encoder block.
            num_classes (int): Number of segmentation classes.
            down_channels_per_block (list of int, optional): Number of intermediate channels
                for each encoder block. Defaults to channels_per_block if None.
            activation_fn (callable, optional): Activation function. Defaults to F.relu.
            convs_per_block (int, optional): Number of convolutional layers per residual block.
                Defaults to 3.
            blocks_per_level (int, optional): Number of residual blocks per decoder level.
                Defaults to 3.
        """
        super(_StitchingDecoder, self).__init__()
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.num_classes = num_classes
        self.activation_fn = activation_fn
        self.convs_per_block = convs_per_block
        self.blocks_per_level = blocks_per_level
        self.down_channels_per_block = (
            down_channels_per_block if down_channels_per_block is not None
            else channels_per_block
        )

        self.name = name
        num_levels = len(channels_per_block)
        num_latents = len(latent_dims)
        start_level = num_latents + 1

        # Predefine residual blocks for each decoder level
        self.decoder_levels = nn.ModuleList()
        for level in range(start_level, num_levels):
            # Input channels to the first residual block depend on the previous level's output
            if level == start_level:
                # Initial decoder_features channels from HierarchicalCore
                previous_out_channels = channels_per_block[num_levels - 1 - num_latents]
            else:
                previous_out_channels = channels_per_block[num_levels - 1 - (level - 1)]
            in_channels = previous_out_channels + channels_per_block[num_levels - 1 - level]
            out_channels = channels_per_block[num_levels - 1 - level]
            down_channels = self.down_channels_per_block[num_levels - 1 - level]

            # Create a sequence of residual blocks for this level
            level_blocks = nn.ModuleList()
            for b in range(blocks_per_level):
                block_in_channels = in_channels if b == 0 else out_channels
                level_blocks.append(
                    ResBlock(block_in_channels, out_channels, down_channels,
                              self.convs_per_block, self.activation_fn)
                )
            self.decoder_levels.append(level_blocks)

        # Final 1x1 convolution to produce logits
        self.final_conv = nn.Conv2d(channels_per_block[0], num_classes, kernel_size=1)

    def forward(self, encoder_features, decoder_features):
        """
        Computes the segmentation logits.

        Args:
            encoder_features (list of torch.Tensor): List of encoder feature tensors with shapes
                [batch, h_i, w_i, c_i].
            decoder_features (torch.Tensor): Decoder feature tensor from HierarchicalCore with shape
                [batch, h, w, c].

        Returns:
            torch.Tensor: Segmentation logits with shape [batch, h, w, num_classes].
        """
        num_levels = len(self.channels_per_block)
        start_level = len(self.latent_dims) + 1

        # Process each decoder level
        for level in range(start_level, num_levels):
            # Upsample decoder features by a factor of 2
            decoder_features = F.interpolate(
                decoder_features, scale_factor=2, mode='bilinear', align_corners=False
            )
            # Concatenate with the corresponding encoder feature (in reverse order)
            encoder_feature = encoder_features[num_levels - 1 - level]
            decoder_features = torch.cat([decoder_features, encoder_feature], dim=1)
            # Apply the pre-defined residual blocks for this level
            for res_block in self.decoder_levels[level - start_level]:
                decoder_features = res_block(decoder_features)

        # Final 1x1 convolution to produce logits
        logits = self.final_conv(decoder_features)
        return logits

def manual_kl_divergence(mu_q, sigma_q, mu_p, sigma_p):
    """Manual KL divergence between two normal distributions."""
    term1 = torch.log(sigma_p / sigma_q)
    term2 = (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2)
    kl = term1 + term2 - 0.5
    return kl
        
class HierarchicalProbUNet(nn.Module):
    """A Hierarchical Probabilistic U-Net in PyTorch."""
    def __init__(self,
                 latent_dims=(1, 1, 1, 1),
                 channels_per_block=None,
                 num_classes=2,
                 down_channels_per_block=None,
                 activation_fn=F.relu,
                 convs_per_block=3,
                 blocks_per_level=3,
                 loss_kwargs=None,
                 in_channels=1,
                 name='HPUNet'):
        super(HierarchicalProbUNet, self).__init__()
        self.name = name
        base_channels = 24
        default_channels_per_block = (
            base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels,
            8 * base_channels, 8 * base_channels, 8 * base_channels, 8 * base_channels
        )
        if channels_per_block is None:
            channels_per_block = default_channels_per_block
        if down_channels_per_block is None and channels_per_block is not None:
            down_channels_per_block = tuple([i // 2 for i in channels_per_block])
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

        # Define submodules
        self._prior = _HierarchicalCore(
            latent_dims=latent_dims,
            channels_per_block=channels_per_block,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            input_channels=in_channels,
            name='prior'
        )
        self._posterior = _HierarchicalCore(
            latent_dims=latent_dims,
            channels_per_block=channels_per_block,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            input_channels=in_channels+1,
            name='posterior'
        )
        self._f_comb = _StitchingDecoder(
            latent_dims=latent_dims,
            channels_per_block=channels_per_block,
            num_classes=num_classes,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name='f_comb'
        )

        # Loss-related utilities for GECO
        if self._loss_kwargs['type'] == 'geco':
            self._moving_average = MovingAverage(
                decay=self._loss_kwargs['decay'], differentiable=True
            )
            self._lagmul = LagrangeMultiplier(rate=self._loss_kwargs['rate'])
        self._cache = None
        self.num_classes = num_classes

    def _build(self, seg, img):
        """Builds the computation graph for training."""
        inputs = (seg, img)
        if self._cache is not None and all(torch.equal(a, b) for a, b in zip(self._cache, inputs)):
            return
        else:
            concat_input = torch.cat([seg, img], dim=1)  # Concatenate along channel dimension
            #print(f'Concat input shape: {concat_input.shape}')
            self._q_sample = self._posterior(concat_input, mean=False)
            self._q_sample_mean = self._posterior(concat_input, mean=True)
            self._p_sample = self._prior(img, mean=False, z_q=None)
            self._p_sample_z_q = self._prior(img, z_q=self._q_sample['used_latents'])
            self._p_sample_z_q_mean = self._prior(img, z_q=self._q_sample_mean['used_latents'])
            self._cache = inputs

    def sample(self, img, mean=False, z_q=None):
        """Sample a segmentation from the prior, given an input image."""
        prior_out = self._prior(img, mean, z_q)
        encoder_features = prior_out['encoder_features']
        decoder_features = prior_out['decoder_features']
        return self._f_comb(encoder_features=encoder_features, decoder_features=decoder_features)

    def reconstruct(self, seg, img, mean=False):
        """Reconstruct a segmentation using the posterior."""
        self._build(seg, img)
        if mean:
            prior_out = self._p_sample_z_q_mean
        else:
            prior_out = self._p_sample_z_q
        encoder_features = prior_out['encoder_features']
        decoder_features = prior_out['decoder_features']
        return self._f_comb(encoder_features=encoder_features, decoder_features=decoder_features)

    def rec_loss(self, seg, img, mask=None, top_k_percentage=None, deterministic=True):
        """Cross-entropy reconstruction loss employed in the ELBO-/ GECO-objective."""
        reconstruction = self.reconstruct(seg, img, mean=False)
        return ce_loss(reconstruction, seg, mask, top_k_percentage, deterministic)

    def kl(self, seg, img):
        """Kullback-Leibler divergence between the posterior and the prior."""
        self._build(seg, img)
        posterior_out = self._q_sample
        prior_out = self._p_sample_z_q

        q_dists = posterior_out['distributions']
        p_dists = prior_out['distributions']

        kl = {}
        for level, (q, p) in enumerate(zip(q_dists, p_dists)):
            mu_q, sigma_q = q.mean, q.stddev
            mu_p, sigma_p = p.mean, p.stddev
            kl_per_pixel = manual_kl_divergence(mu_q, sigma_q, mu_p, sigma_p)
            kl_per_instance = torch.sum(kl_per_pixel, dim=[1, 2])
            kl[level] = torch.mean(kl_per_instance)
        return kl

    def loss(self, seg, img, mask):
        """The full training objective, either ELBO or GECO."""

        summaries = {}
        top_k_percentage = self._loss_kwargs['top_k_percentage']
        deterministic = self._loss_kwargs['deterministic_top_k']
            
        #print(f"Seg shape: {seg.shape} ===== Img shape: {img.shape} ===== Mask shape: {mask.shape}")
        rec_loss = self.rec_loss(seg, img, mask, top_k_percentage, deterministic)

        mask = None # No mask for now 

        kl_dict = self.kl(seg, img)
        kl_sum = torch.sum(torch.stack([kl for _, kl in kl_dict.items()], dim=0))

        summaries['rec_loss_mean'] = rec_loss['mean']
        summaries['rec_loss_sum'] = rec_loss['sum']
        summaries['kl_sum'] = kl_sum
        for level, kl in kl_dict.items():
            summaries[f'kl_{level}'] = kl

        if self._loss_kwargs['type'] == 'elbo':
            loss = rec_loss['sum'] + self._loss_kwargs['beta'] * kl_sum
            summaries['elbo_loss'] = loss

        elif self._loss_kwargs['type'] == 'geco':
            ma_rec_loss = self._moving_average(rec_loss['sum'])
            mask_sum_per_instance = torch.sum(rec_loss['mask'], dim=1)  # sum over the flattened pixels
            num_valid_pixels = torch.mean(mask_sum_per_instance)
            reconstruction_threshold = self._loss_kwargs['kappa'] * num_valid_pixels

            rec_constraint = ma_rec_loss - reconstruction_threshold
            lagmul = self._lagmul(rec_constraint)
            loss = lagmul * rec_constraint + kl_sum

            summaries['geco_loss'] = loss
            summaries['ma_rec_loss_mean'] = ma_rec_loss / num_valid_pixels
            summaries['num_valid_pixels'] = num_valid_pixels
            summaries['lagmul'] = lagmul
        else:
            raise NotImplementedError(f"Loss type {self._loss_kwargs['type']} not implemented!")

        return dict(supervised_loss=loss, summaries=summaries)

if __name__ == '__main__':
    hpu_net = HierarchicalProbUNet()