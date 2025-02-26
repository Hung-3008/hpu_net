#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

# Import your dataset and augmentation
from dataset_ACDC import ACDCdataset, RandomGenerator

# Import the model (assumed to be defined in hierarchical_probabilistic_unet/model.py)
from model import HierarchicalProbUNet

def train_epoch(model, dataloader, optimizer, device, writer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, sample in enumerate(dataloader):
        # Get the image and label from the sample.
        # In your dataset, 'image' is shape (B, 1, H, W) and 'label' is (B, H, W) (class indices).
        img = sample['image'].to(device)
        label = sample['label'].to(device)  # integer labels

        optimizer.zero_grad()
        # The model's loss() method expects seg (label) and img.
        loss_dict = model.loss(label, img, mask=None)
        loss = loss_dict['supervised_loss']
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            # Log batch-level training loss
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Training/BatchLoss', loss.item(), global_step)
            
            # Log additional loss components if available
            for k, v in loss_dict.items():
                if k != 'supervised_loss' and isinstance(v, torch.Tensor):
                    writer.add_scalar(f'Training/{k}', v.item(), global_step)
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def main():
    # -------------------------------
    # Configuration (adjust as needed)
    # -------------------------------
    base_dir = "datasets/ACDC"     # Path where the ACDC .npz files are stored.
    list_dir = "datasets/ACDC/lists_ACDC"     # Path to the text files listing the cases.
    split = "train"                     # or "valid"
    output_size = (128, 128)            # Target output size.
    batch_size = 2
    num_epochs = 50
    learning_rate = 1e-4
    
    # Create logs directory for TensorBoard
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", f"hpunet_run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print(f"TensorBoard logs will be saved to {log_dir}")
    print(f"View them by running: tensorboard --logdir={log_dir}")

    # -------------------------------
    # Create the Dataset and DataLoader
    # -------------------------------
    transform = RandomGenerator(output_size)
    train_dataset = ACDCdataset(base_dir, list_dir, split, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # -------------------------------
    # Set up device and model
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalProbUNet(
        latent_dims=(4, 4, 4),                # 3 latent scales, each with 4 channels
        channels_per_block=(64, 128, 256, 512, 512),  # 5 encoder levels
        num_classes=4,                        # For binary segmentation (e.g. background vs. myocardium)
        down_channels_per_block=(32, 64, 128, 256, 256), # Intermediate channel counts
        activation_fn=F.relu,
        convs_per_block=3,
        blocks_per_level=3,
        loss_kwargs={
            'type': 'geco',
            'top_k_percentage': 0.02,
            'deterministic_top_k': False,
            'kappa': 0.05,
            'decay': 0.99,
            'rate': 1e-2,
            'beta': 1.0
        }
    ).to(device)

    # Log model graph
    sample_input = torch.randn(1, 1, output_size[0], output_size[1]).to(device)
    try:
        writer.add_graph(model, sample_input)
    except Exception as e:
        print(f"Failed to add model graph to TensorBoard: {e}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # -------------------------------
    # Training Loop
    # -------------------------------
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, device, writer, epoch)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Log epoch-level metrics
        writer.add_scalar('Training/EpochLoss', avg_loss, epoch)
        writer.add_scalar('Training/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log sample predictions periodically
        if epoch % 5 == 0:
            # Get a sample batch
            with torch.no_grad():
                for sample in train_loader:
                    img = sample['image'].to(device)
                    label = sample['label'].to(device)
                    
                    # Generate a prediction (assuming the model has a predict method)
                    try:
                        pred = model.sample_predictions(img)
                        
                        # Log example images
                        writer.add_images('Images/Input', img, epoch)
                        
                        # For visualization, convert label and prediction to same format
                        label_vis = F.one_hot(label, num_classes=4).permute(0, 3, 1, 2).float()
                        writer.add_images('Images/GroundTruth', label_vis, epoch)
                        
                        if pred is not None:
                            writer.add_images('Images/Prediction', pred, epoch)
                    except Exception as e:
                        print(f"Failed to log prediction images: {e}")
                    break  # Only process one batch
        
        # Save checkpoint
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            checkpoint_dir = os.path.join("checkpoints", timestamp)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"hpunet_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    main()