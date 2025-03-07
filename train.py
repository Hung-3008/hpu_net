import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import logging
import argparse
import yaml
from torch.nn import functional as F


from dataset_LIDC import LIDC_IDRI  
from model import HierarchicalProbUNet 

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Hierarchical Probabilistic U-Net")
    parser.add_argument('--config', type=str, default='configs/config.yml', 
                        help='Path to YAML configuration file')
    parser.add_argument('--data_dir', type=str, default='LIDC', 
                        help='Path to dataset (overrides config if provided)')
    parser.add_argument('--exp_dir', type=str, default='experiments/hpu_net', 
                        help='Experiment directory (overrides config if provided)')
    parser.add_argument('--cuda', type=str, default='0', 
                        help='CUDA device ID (overrides config if provided)')
    return parser.parse_args()


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=0.01)  # Smaller std than Kaiming
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



def train_epoch(model, train_loader, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    for images, labels, _ in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Add channel dimension
        mask = torch.ones_like(labels).to(device)  # Full mask for simplicity
        
        if torch.isnan(images).any() or torch.isnan(labels).any():
            print("NaN detected in inputs")
            logging.warning("NaN detected in inputs")
            continue

        optimizer.zero_grad()
        loss_dict = model.loss(labels, images, mask)
        loss = loss_dict['supervised_loss']
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, device, n_batches):
    """Validate the model."""
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(val_loader):
            if batch_idx >= n_batches:
                break
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            mask = torch.ones_like(labels).to(device)
            loss_dict = model.loss(labels, images, mask)
            total_val_loss += loss_dict['supervised_loss'].item()
    return total_val_loss / n_batches

def train(cf):
    """Main training function."""
    # Setup logging
    if not os.path.exists(cf['exp_dir']):
        os.makedirs(cf['exp_dir'])
    
    log_file = os.path.join(cf['exp_dir'], 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting training with config: %s", cf)

    # Device setup
    device = torch.device(f"cuda:{cf['cuda_visible_devices']}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data loaders
    train_dataset = LIDC_IDRI(cf['data_dir'], split='train', transform=None)
    val_dataset = LIDC_IDRI(cf['data_dir'], split='val', transform=None)
    train_loader = DataLoader(train_dataset, batch_size=cf['batch_size'], 
                              shuffle=True, num_workers=cf['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cf['batch_size'], 
                            shuffle=False, num_workers=cf['num_workers'])
    logging.info(f"Loaded {len(train_dataset)} train and {len(val_dataset)} val samples")

    # Model and optimizer
    model = HierarchicalProbUNet(
        latent_dims=tuple(cf['latent_dims']),
        channels_per_block=cf['channels_per_block'],
        num_classes=cf['num_classes'],
        activation_fn=cf['activation_fn'],
        convs_per_block=cf['convs_per_block'],
        blocks_per_level=cf['blocks_per_level'],
        loss_kwargs=cf['loss_kwargs'],
        in_channels=1  # Assuming grayscale images
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cf['learning_rate'])
    model.apply(init_weights)
    logging.info("Model and optimizer initialized")

    # Training loop
    for epoch in range(cf['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device, cf['validation']['n_batches'])
        logging.info(f"Epoch {epoch+1}/{cf['num_epochs']}: Train Loss: {train_loss:.4f}, "
                     f"Val Loss: {val_loss:.4f}")

        if (epoch + 1) % cf['save_every_n_epochs'] == 0:
            checkpoint_path = os.path.join(cf['exp_dir'], f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    args = parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cf = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    activation_map = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    }
    cf['activation_fn'] = activation_map[cf['activation_fn']]

    train(cf)