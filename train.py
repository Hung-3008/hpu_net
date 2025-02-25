#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import HierarchicalProbUNet
from dataset_ACDC import ACDCdataset, RandomGenerator 

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up the training transform (data augmentation and resizing)
    transform = RandomGenerator(output_size=(args.height, args.width))
    
    # Create the training dataset and DataLoader.
    train_dataset = ACDCdataset(base_dir=args.data_dir, 
                                list_dir=args.list_dir, 
                                split="train", 
                                transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Instantiate the Hierarchical Probabilistic U-Net.
    # For the prior core, we assume input is a single channel image.
    # For the posterior, the input is the concatenation of image and segmentation mask.
    model = HierarchicalProbUNet(
        latent_dims=tuple(args.latent_dims),
        channels_per_block=args.channels_per_block,
        num_classes=args.num_classes,
        prior_in_channels=1,
        posterior_in_channels=1 + args.num_classes  # Image (1) + one-hot segmentation (num_classes)
    ).to(device)

    # Use Adam optimizer.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop.
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_idx, sample in enumerate(train_loader):
            # sample is a dict with keys: 'image', 'label', and 'case_name'.
            image = sample['image'].to(device)  # expected shape: (B, 1, H, W)
            label = sample['label'].to(device)  # expected shape: (B, H, W) or (B, num_classes, H, W)
            print(f"Image shape: {image.shape}, Label shape: {label.shape}")
            
            # Convert labels to one-hot encoding
            if len(label.shape) == 3:  # If label is not already one-hot
                label_onehot = torch.nn.functional.one_hot(label.long(), num_classes=args.num_classes)
                label_onehot = label_onehot.permute(0, 3, 1, 2).float()
            else:
                label_onehot = label
                
            print(f"Label one-hot shape: {label_onehot.shape}")
            print(f"Label one-hot unique values: {label_onehot.unique()}")
            
            optimizer.zero_grad()

            # Compute loss. The model.loss() method expects (segmentation, image, mask)
            loss_dict = model.loss(label_onehot, image, mask=None)
            loss = loss_dict['supervised_loss']

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")

        # Optionally, save a checkpoint.
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Hierarchical Probabilistic U-Net on ACDC dataset")
    parser.add_argument("--data_dir", type=str, default="./datasets/ACDC", 
                        help="Base directory for ACDC data")
    parser.add_argument("--list_dir", type=str, default="./datasets/ACDC/lists_ACDC", 
                        help="Directory containing split list files (e.g. train.txt)")
    parser.add_argument("--save_dir", type=str, default="checkpoints", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=10, 
                        help="Interval (in batches) for printing training status")
    parser.add_argument("--save_interval", type=int, default=10, 
                        help="Interval (in epochs) for saving checkpoints")
    parser.add_argument("--height", type=int, default=256, help="Output image height")
    parser.add_argument("--width", type=int, default=256, help="Output image width")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # Model-specific hyperparameters.
    parser.add_argument("--latent_dims", type=int, nargs='+', default=[3, 2, 1],
                        help="List of latent dimensions at each scale")
    parser.add_argument("--channels_per_block", type=int, nargs='+', 
                        default=[24, 48, 96, 192, 192, 192, 192, 192],
                        help="Number of channels per block (for encoder)")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of segmentation classes")
    parser.add_argument("--posterior_in_channels", type=int, default=None, 
                        help="Number of input channels for posterior (defaults to 1 + num_classes)")

    args = parser.parse_args()
    
    # Set the posterior_in_channels if not explicitly provided
    if args.posterior_in_channels is None:
        args.posterior_in_channels = 1 + args.num_classes
        
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
