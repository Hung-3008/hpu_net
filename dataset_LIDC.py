import torch
import os
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class LIDC_IDRI(Dataset):
    def __init__(self, dataset_location, split='train', transform=None):
        """
        Initialize the LIDC-IDRI dataset with a directory structure containing images and ground truth masks.

        Args:
            dataset_location (str): Path to the root directory (e.g., 'LIDC/').
            split (str): Dataset split to use ('train', 'val', or 'test'). Default: 'train'.
            transform (callable, optional): Optional transform to apply to images and labels.
        """
        self.transform = transform
        self.split = split
        self.image_dir = os.path.join(dataset_location, split, 'images')
        self.gt_dir = os.path.join(dataset_location, split, 'gt')

        # List to store (image_path, gt_path, subfolder) tuples
        self.data_pairs = []

        # Ensure directories exist
        if not os.path.exists(self.image_dir) or not os.path.exists(self.gt_dir):
            raise FileNotFoundError(f"Image or GT directory not found for split '{split}'")

        # Iterate through subfolders in the images directory
        for subfolder in os.listdir(self.image_dir):
            image_subfolder_path = os.path.join(self.image_dir, subfolder)
            gt_subfolder_path = os.path.join(self.gt_dir, subfolder)

            # Check if the subfolder exists in both image and gt directories
            if not os.path.isdir(image_subfolder_path) or not os.path.isdir(gt_subfolder_path):
                continue

            # Get all image files in the subfolder
            missing = 0
            image_files = [f for f in os.listdir(image_subfolder_path) if f.endswith('.png')]
            for image_file in image_files:
                image_path = os.path.join(image_subfolder_path, image_file)
                # Assume the ground truth file has the same name
                imge_name = image_file.split('.png')[0]
                gt_file = imge_name + '_l0.png'

                gt_path = os.path.join(gt_subfolder_path, gt_file)

                if os.path.exists(gt_path):
                    self.data_pairs.append((image_path, gt_path, subfolder))
                else:
                    print(f"Warning: No matching GT file for {image_path}")
                    missing += 1
        

        if not self.data_pairs:
            raise ValueError(f"No valid image-GT pairs found in split '{split}'")
        print(f"Found {missing} missing GT files in split '{split}'")
        print(f"Found {len(self.data_pairs)} image-GT pairs in split '{split}'")

    def __getitem__(self, index):
        """
        Fetch an item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: (image, label, series_uid)
        """
        # Get the paths for the image and ground truth
        image_path, gt_path, series_uid = self.data_pairs[index]

        # Load the image and ground truth using PIL and resize them to 256x256
        pil_image = Image.open(image_path)
        pil_image = pil_image.resize((256, 256), resample=Image.BILINEAR)
        image = np.array(pil_image)  # Shape: (H, W) for grayscale

        pil_label = Image.open(gt_path)
        pil_label = pil_label.resize((256, 256), resample=Image.NEAREST)
        label = np.array(pil_label)  # Shape: (H, W)

        # Normalize image to [0, 1]
        if image.max() > 1:
            image = image / 255.0
        image = image.astype(np.float32)

        # Ensure label is in [0, 1] (binary or multi-class)
        if label.max() > 1:
            label = label / 255.0
        label = label.astype(np.float32)

        # Add channel dimension to image: (H, W) -> (1, H, W)
        image = np.expand_dims(image, axis=0)

        # Apply transforms if provided
        if self.transform is not None:
            # Stack image and label for joint transforms (e.g., random rotation)
            stacked = np.concatenate([image, np.expand_dims(label, axis=0)], axis=0)  # Shape: (2, H, W)
            transformed = self.transform(torch.from_numpy(stacked))
            image, label = transformed[0:1], transformed[1]  # Split back into image and label

        # Convert to PyTorch tensors if not already transformed
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()  # Shape: (1, H, W)
            label = torch.from_numpy(label).float()  # Shape: (H, W)

        return image, label, series_uid

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.data_pairs)

# Example usage
if __name__ == "__main__":
    # Define dataset location and split
    dataset_location = "LIDC"
    train_dataset = LIDC_IDRI(dataset_location, split='train', transform=None)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    # Iterate through a batch
    for batch_idx, (images, labels, series_uids) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"Image shape: {images.shape}")  # Should be (batch_size, 1, H, W)
        print(f"Label shape: {labels.shape}")  # Should be (batch_size, H, W)
        print(f"Series UIDs: {series_uids}")
        break