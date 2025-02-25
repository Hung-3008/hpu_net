import unittest
import torch
import torch.nn.functional as F
import numpy as np

from model import HierarchicalProbUNet

# Test configuration.
_NUM_CLASSES = 2
_BATCH_SIZE = 2
_SPATIAL_SHAPE = [32, 32]  # Height, Width
# For PyTorch, image shape is (B, C, H, W). In the original, images are grayscale.
_IMAGE_SHAPE = [_BATCH_SIZE, 1, _SPATIAL_SHAPE[0], _SPATIAL_SHAPE[1]]
# Segmentation maps have _NUM_CLASSES channels.
_SEGMENTATION_SHAPE = [_BATCH_SIZE, _NUM_CLASSES, _SPATIAL_SHAPE[0], _SPATIAL_SHAPE[1]]
_CHANNELS_PER_BLOCK = [5, 7, 9, 11, 13]
# Compute bottleneck spatial size as in the original:
# _BOTTLENECK_SIZE = _SPATIAL_SHAPE[0] // 2**(len(_CHANNELS_PER_BLOCK)-1)
_BOTTLENECK_SIZE = _SPATIAL_SHAPE[0] // (2 ** (len(_CHANNELS_PER_BLOCK) - 1))  # 32 // 16 = 2
_LATENT_DIMS = [3, 2, 1]

def _get_inputs():
    """Creates dummy inputs for testing.
    
    Returns:
      img: Tensor of shape _IMAGE_SHAPE (grayscale image).
      seg: Tensor of shape _SEGMENTATION_SHAPE (segmentation, one-hot or continuous).
    """
    torch.manual_seed(0)
    img = torch.randn(_IMAGE_SHAPE)
    seg = torch.randn(_SEGMENTATION_SHAPE)
    return img, seg

class HierarchicalProbUNetTest(unittest.TestCase):

    def test_shape_of_sample(self):
        # Instantiate the model.
        # For the prior, image channels = 1; for the posterior, segmentation+image = 1+2 = 3.
        model = HierarchicalProbUNet(latent_dims=_LATENT_DIMS,
                                     channels_per_block=_CHANNELS_PER_BLOCK,
                                     num_classes=_NUM_CLASSES,
                                     prior_in_channels=1,
                                     posterior_in_channels=3)
        img, _ = _get_inputs()
        sample = model.sample(img)
        # Expect segmentation output: shape (B, num_classes, H, W)
        self.assertEqual(sample.size(), torch.Size(_SEGMENTATION_SHAPE))

    def test_shape_of_reconstruction(self):
        model = HierarchicalProbUNet(latent_dims=_LATENT_DIMS,
                                     channels_per_block=_CHANNELS_PER_BLOCK,
                                     num_classes=_NUM_CLASSES,
                                     prior_in_channels=1,
                                     posterior_in_channels=3)
        img, seg = _get_inputs()
        reconstruction = model.reconstruct(seg, img)
        self.assertEqual(reconstruction.size(), torch.Size(_SEGMENTATION_SHAPE))

    def test_shapes_in_prior(self):
        model = HierarchicalProbUNet(latent_dims=_LATENT_DIMS,
                                     channels_per_block=_CHANNELS_PER_BLOCK,
                                     num_classes=_NUM_CLASSES,
                                     prior_in_channels=1,
                                     posterior_in_channels=3)
        img, _ = _get_inputs()
        # In our conversion, the prior core is accessible via model.prior.
        # Call with default mean=False.
        prior_out = model.prior(img, mean=False)
        distributions = prior_out['distributions']
        latents = prior_out['used_latents']
        encoder_features = prior_out['encoder_features']
        decoder_features = prior_out['decoder_features']

        # Check that the number of latent distributions equals the number of latent scales.
        self.assertEqual(len(distributions), len(_LATENT_DIMS))

        # Check shapes of latent scales.
        # Expected latent spatial shape = _BOTTLENECK_SIZE * 2**level, and tensor shape: (B, latent_dim, H, W)
        for level, latent_dim in enumerate(_LATENT_DIMS):
            expected_spatial = _BOTTLENECK_SIZE * (2 ** level)
            expected_shape = (_BATCH_SIZE, latent_dim, expected_spatial, expected_spatial)
            self.assertEqual(latents[level].size(), torch.Size(expected_shape))

        # Check encoder features.
        # At level i, expected spatial size = _SPATIAL_SHAPE[0] // (2**i), and channels = _CHANNELS_PER_BLOCK[i].
        for level, channels in enumerate(_CHANNELS_PER_BLOCK):
            spatial = _SPATIAL_SHAPE[0] // (2 ** level)
            expected_shape = (_BATCH_SIZE, channels, spatial, spatial)
            self.assertEqual(encoder_features[level].size(), torch.Size(expected_shape))

        # Check decoder features.
        # In the original, start_level = len(_LATENT_DIMS). The expected spatial size:
        start_level = len(_LATENT_DIMS)
        expected_spatial = _BOTTLENECK_SIZE * (2 ** start_level)
        # Expected channels from the reversed list:
        expected_channels = _CHANNELS_PER_BLOCK[::-1][start_level]
        expected_shape = (_BATCH_SIZE, expected_channels, expected_spatial, expected_spatial)
        self.assertEqual(decoder_features.size(), torch.Size(expected_shape))

    def test_shape_of_kl(self):
        model = HierarchicalProbUNet(latent_dims=_LATENT_DIMS,
                                     channels_per_block=_CHANNELS_PER_BLOCK,
                                     num_classes=_NUM_CLASSES,
                                     prior_in_channels=1,
                                     posterior_in_channels=3)
        img, seg = _get_inputs()
        kl_dict = model.kl(img, seg)
        # Check that KL divergence is computed for each latent level.
        self.assertEqual(len(kl_dict), len(_LATENT_DIMS))

if __name__ == '__main__':
    unittest.main()