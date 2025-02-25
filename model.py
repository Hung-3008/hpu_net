import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

def res_block(x, out_channels, out_channels_down=None, convs_per_block=3):
    # Example PyTorch equivalent of unet_utils.res_block
    if out_channels_down is None:
        out_channels_down = out_channels
    conv = nn.Sequential(
        nn.Conv2d(x.shape[1], out_channels_down, kernel_size=3, padding=1),
        nn.ReLU(),
        *[nn.Sequential(
            nn.Conv2d(out_channels_down, out_channels_down, kernel_size=3, padding=1),
            nn.ReLU(),
        ) for _ in range(convs_per_block - 2)],
        nn.Conv2d(out_channels_down, out_channels, kernel_size=3, padding=1),
    )
    return conv(x)

def resize_down(x, scale=2):
    return F.avg_pool2d(x, kernel_size=scale, stride=scale)

def resize_up(x, scale=2):
    return F.interpolate(x, scale_factor=scale, mode='nearest')

class _HierarchicalCore(nn.Module):
    def __init__(self, latent_dims, channels_per_block, down_channels_per_block=None,
                 activation_fn=F.relu, convs_per_block=3, blocks_per_level=3):
        super().__init__()
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.down_channels_per_block = down_channels_per_block or channels_per_block
        self.activation_fn = activation_fn
        self.convs_per_block = convs_per_block
        self.blocks_per_level = blocks_per_level
        # Simple example, no explicit layers built here; see res_block usage below

    def forward(self, x, mean=False, z_q=None):
        encoder_outputs = []
        distributions, used_latents = [], []
        num_levels = len(self.channels_per_block)
        num_latent_levels = len(self.latent_dims)
        if isinstance(mean, bool):
            mean = [mean] * num_latent_levels

        # Encoder
        feats = x
        for lvl in range(num_levels):
            for _ in range(self.blocks_per_level):
                feats = res_block(feats, self.channels_per_block[lvl],
                                  self.down_channels_per_block[lvl], self.convs_per_block)
            encoder_outputs.append(feats)
            if lvl < num_levels - 1:
                feats = resize_down(feats, 2)

        # Decoder with latents
        decoder_features = encoder_outputs[-1]
        for lvl in range(num_latent_levels):
            mu_logsigma = nn.Conv2d(decoder_features.shape[1], 2*self.latent_dims[lvl],
                                    kernel_size=1)(decoder_features)
            mu, logsigma = mu_logsigma[:, :self.latent_dims[lvl]], mu_logsigma[:, self.latent_dims[lvl]:]
            dist = Independent(Normal(loc=mu, scale=torch.exp(logsigma)), 1)
            distributions.append(dist)
            if z_q is not None:
                z = z_q[lvl]
            elif mean[lvl]:
                z = dist.mean
            else:
                z = dist.sample()
            used_latents.append(z)
            decoder_lo = torch.cat([z, decoder_features], dim=1)
            decoder_hi = resize_up(decoder_lo, 2)
            decoder_features = torch.cat([decoder_hi, encoder_outputs[-2 - lvl]], dim=1)
            for _ in range(self.blocks_per_level):
                out_ch = self.channels_per_block[-2 - lvl]
                out_ch_down = self.down_channels_per_block[-2 - lvl]
                decoder_features = res_block(decoder_features, out_ch, out_ch_down, self.convs_per_block)

        return {
            'decoder_features': decoder_features,
            'encoder_features': encoder_outputs,
            'distributions': distributions,
            'used_latents': used_latents
        }

class _StitchingDecoder(nn.Module):
    def __init__(self, latent_dims, channels_per_block, num_classes,
                 down_channels_per_block=None, convs_per_block=3, blocks_per_level=3):
        super().__init__()
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.num_classes = num_classes
        self.down_channels_per_block = down_channels_per_block or channels_per_block
        self.convs_per_block = convs_per_block
        self.blocks_per_level = blocks_per_level
        self.out_conv = nn.Conv2d(channels_per_block[0], num_classes, kernel_size=1)

    def forward(self, encoder_features, decoder_features):
        start_level = len(self.latent_dims) + 1
        num_levels = len(self.channels_per_block)
        feats = decoder_features
        for lvl in range(start_level, num_levels):
            feats = resize_up(feats, 2)
            feats = torch.cat([feats, encoder_features[-1 - lvl]], dim=1)
            for _ in range(self.blocks_per_level):
                out_ch = self.channels_per_block[-1 - lvl]
                out_ch_down = self.down_channels_per_block[-1 - lvl]
                feats = res_block(feats, out_ch, out_ch_down, self.convs_per_block)
        return self.out_conv(feats)

class HierarchicalProbUNet(nn.Module):
    def __init__(self, latent_dims=(1,1,1,1), channels_per_block=None, num_classes=2,
                 down_channels_per_block=None, convs_per_block=3, blocks_per_level=3):
        super().__init__()
        if channels_per_block is None:
            channels_per_block = (24, 48, 96, 192, 192, 192, 192, 192)
        if down_channels_per_block is None:
            down_channels_per_block = tuple(ch//2 for ch in channels_per_block)

        self.prior = _HierarchicalCore(latent_dims, channels_per_block, down_channels_per_block,
                                       convs_per_block=convs_per_block, blocks_per_level=blocks_per_level)
        self.posterior = _HierarchicalCore(latent_dims, channels_per_block, down_channels_per_block,
                                           convs_per_block=convs_per_block, blocks_per_level=blocks_per_level)
        self.f_comb = _StitchingDecoder(latent_dims, channels_per_block, num_classes,
                                        down_channels_per_block, convs_per_block, blocks_per_level)

    def forward(self, img):
        out = self.prior(img)
        return self.f_comb(out['encoder_features'], out['decoder_features'])

    def sample(self, img, mean=False, z_q=None):
        prior_out = self.prior(img, mean, z_q)
        return self.f_comb(prior_out['encoder_features'], prior_out['decoder_features'])

    def reconstruct(self, seg, img, mean=False):
        # Posterior forward
        post_out = self.posterior(torch.cat([seg, img], dim=1))
        # Use posterior latents in prior
        if mean:
            q_means = [d.mean for d in post_out['distributions']]
            prior_out = self.prior(img, z_q=q_means)
        else:
            prior_out = self.prior(img, z_q=post_out['used_latents'])
        return self.f_comb(prior_out['encoder_features'], prior_out['decoder_features'])