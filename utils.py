import torch
import matplotlib.pyplot as plt
import numpy as np 


def visualize_medical_data(images, masks, figsize=(16, 8)):
    """
    Visualize medical images and their corresponding segmentation masks.
    
    Parameters:
    -----------
    images : torch.Tensor
        Batch of medical images with shape (batch_size, channels, height, width)
    masks : torch.Tensor
        Batch of segmentation masks with shape (batch_size, height, width)
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for further customization if needed
    """
    batch_size = min(images.shape[0], 2)  # Display up to 2 images
    
    fig, axes = plt.subplots(batch_size, 4, figsize=figsize)
    
    # Handle the case when batch_size is 1
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):  
        # Squeeze the channel dimension for images
        img = images[i].squeeze().numpy()  # Convert (1, H, W) to (H, W)
        mask = masks[i].numpy()

        # Display the original image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # Display the mask
        axes[i, 1].imshow(mask, cmap='nipy_spectral')
        axes[i, 1].set_title(f'Segmentation Mask {i+1}')
        axes[i, 1].axis('off')
        
        # Display the image with mask overlay
        axes[i, 2].imshow(img, cmap='gray')
        axes[i, 2].imshow(mask, alpha=0.5, cmap='nipy_spectral')
        axes[i, 2].set_title(f'Overlay {i+1}')
        axes[i, 2].axis('off')
        
        # Display unique values in the mask
        unique_values = np.unique(mask)
        axes[i, 3].text(0.1, 0.5, f"Unique classes: {unique_values}", fontsize=12)
        axes[i, 3].axis('off')

    plt.tight_layout()
    
    # Print unique values in all masks
    all_unique_values = np.unique(masks.numpy())
    print(f"All unique class labels in the batch: {all_unique_values}")
    
    return fig