############ train_acdc.py
import time
from random import randrange

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import HPUNet

def train_model(args, model, dataloader, criterion, optimizer, lr_scheduler, writer, device='cpu', val_dataloader=None, start_time=None): 
    history = {
        'training_time(min)': None
    }

    if val_dataloader is not None:
        val_minibatches = len(val_dataloader)

    def record_history(idx, loss_dict, type='train'):
        prefix = 'Minibatch Training ' if type == 'train' else 'Mean Validation '

        loss_per_pixel = loss_dict['loss'].item() / args.pixels
        reconstruction_per_pixel = loss_dict['reconstruction_term'].item() / args.pixels
        kl_term_per_pixel = loss_dict['kl_term'].item() / args.pixels
        kl_per_pixel = [ loss_dict['kls'][v].item() / args.pixels for v in range(args.latent_num) ]

        # Total Loss
        _dict = {   
            'total': loss_per_pixel,
            'kl term': kl_term_per_pixel, 
            'reconstruction': reconstruction_per_pixel  
        }
        writer.add_scalars(prefix + 'Loss Curve', _dict, idx)

        # KL Term Decomposition
        _dict = { 'sum': sum(kl_per_pixel) }
        _dict.update({ 'scale {}'.format(v): kl_per_pixel[v] for v in range(args.latent_num) })
        writer.add_scalars(prefix + 'Loss Curve (K-L)', _dict, idx)

        # Coefficients
        if type == 'train':
            if args.loss_type.lower() == 'elbo':
                writer.add_scalar('Beta', criterion.beta_scheduler.beta, idx)
            elif args.loss_type.lower() == 'geco':
                lamda = criterion.log_inv_function(criterion.log_lamda).item()
                writer.add_scalar('Lagrange Multiplier', lamda, idx)
                writer.add_scalar('Beta', 1/(lamda+1e-20), idx)

    # Prepare a batch of validation images and labels for visualization.
    # Note: Each item in the dataloader is a dict. We assume the keys are 'image' and 'label'.
    val_batch = next(iter(val_dataloader))
    # For visualization, select a subset and extract the tensors.
    val_images = val_batch['image'][:16]

    # Ensure validation images have the correct number of channels (1)
    if val_images.ndim == 3:  # If missing channel dimension
        val_images = val_images.unsqueeze(1)  # Add channel dimension
    elif val_images.shape[1] != 1:  # If wrong number of channels
        # Take first channel or average all channels
        val_images = val_images[:, 0:1, :, :]  # Take first channel only

    val_truths = val_batch['label'][:16]
    truth_grid = make_grid(val_truths, nrow=4, pad_value=val_truths.min().item())
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(truth_grid[0])
    ax.set_axis_off()
    fig.tight_layout()
    writer.add_figure('Validation Images / Ground Truth', fig)
    val_images_selection = val_images.to(device).float() 
    
    last_time_checkpoint = start_time
    for e in range(args.epochs):
        for mb, sample in enumerate(tqdm(dataloader)):
            idx = e * len(dataloader) + mb + 1

            # Set to training mode
            criterion.train()
            model.train()
            model.zero_grad()

            # Extract images and labels from the dictionary
            images = sample['image'].to(device).float()
            truths = sample['label'].to(device).float()

            # If needed, squeeze or adjust dimensions to match your model's expected input
            # (For example, if truths are provided with an extra dimension)
            truths = truths.squeeze(dim=1)
            truths_unsqueezed = truths.unsqueeze(1)
            # Forward pass: if your model expects a target for posterior network training, pass truths
            preds, infodicts = model(images, truths_unsqueezed)
            preds, infodict = preds[:,0], infodicts[0]

            # Calculate Loss
            loss = criterion(preds, truths, kls=infodict['kls'], lr=lr_scheduler.get_last_lr()[0])

            # Backpropagate and update weights
            loss.backward()
            optimizer.step()
            
            # Update beta scheduler if using ELBO loss
            if args.loss_type.lower() == 'elbo':
                criterion.beta_scheduler.step()

            # Record training history
            loss_dict = criterion.last_loss.copy()
            loss_dict.update({'kls': infodict['kls']})
            record_history(idx, loss_dict)
            
            # Validation periodically
            if idx % args.val_period == 0 and val_dataloader is not None:
                criterion.eval()
                model.eval()
                with torch.no_grad():
                    # Forward pass on the validation images; here we assume the posterior branch is not used
                    val_preds, _ = model(val_images_selection.float())  # Add .float() here to match model precision
                    val_preds = val_preds[:,0]
                    
                    out_grid = make_grid(val_preds, nrow=4, pad_value=val_preds.min().item())
                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.imshow(out_grid[0].cpu())
                    ax.set_axis_off()
                    fig.tight_layout()
                    writer.add_figure('Validation Images / Prediction', fig, idx)

                # Calculate validation loss
                mean_val_loss = torch.zeros(1, device=device)
                mean_val_reconstruction_term = torch.zeros(1, device=device)
                mean_val_kl_term = torch.zeros(1, device=device)
                mean_val_kl = torch.zeros(args.latent_num, device=device)

                with torch.no_grad():
                    for _, val_sample in enumerate(val_dataloader):
                        val_images = val_sample['image'].to(device).float()  # This is already float
                        
                        # Explicitly select first channel only or reshape to 1 channel
                        if val_images.shape[1] != 1:
                            print('Warning: Validation images have more than 1 channel. Taking first channel only.')
                            print('Shape before:', val_images.shape)
                            continue
                            #val_images = val_images[:, 0:1, :, :]  # Take only the first channel
                        
                        val_truths = val_sample['label'].to(device).float()
                        val_truths = val_truths.squeeze(dim=1)
                        val_truths_unsqueezed = val_truths.unsqueeze(1)

                        val_preds, val_infodicts = model(val_images, val_truths_unsqueezed)  # All tensors are now float
                        val_preds, val_infodict = val_preds[:,0], val_infodicts[0]

                        loss = criterion(val_preds, val_truths, kls=val_infodict['kls'])
                        mean_val_loss += loss
                        mean_val_reconstruction_term += criterion.last_loss['reconstruction_term']
                        mean_val_kl_term += criterion.last_loss['kl_term']
                        mean_val_kl += val_infodict['kls']
                    
                    mean_val_loss /= val_minibatches
                    mean_val_reconstruction_term /= val_minibatches
                    mean_val_kl_term /= val_minibatches
                    mean_val_kl /= val_minibatches

                loss_dict = {
                    'loss': mean_val_loss,
                    'reconstruction_term': mean_val_reconstruction_term,
                    'kl_term': mean_val_kl_term,
                    'kls': mean_val_kl
                }
                record_history(idx, loss_dict, type='val')
        
        # End of epoch: record time and adjust learning rate
        time_checkpoint = time.time()
        epoch_time = (time_checkpoint - last_time_checkpoint) / 60
        total_time = (time_checkpoint - start_time) / 60
        print(f'Epoch {e+1}/{args.epochs} done in {epoch_time:.1f} minutes. Total time: {total_time:.1f} minutes.')
        last_time_checkpoint = time_checkpoint
        
        # Save checkpoint periodically
        if (e+1) % args.save_period == 0 and (e+1) != args.epochs:
            torch.save(model.state_dict(), f'{args.output_dir}/{args.stamp}/model{e+1}.pth')
            torch.save(criterion.state_dict(), f'{args.output_dir}/{args.stamp}/loss{e+1}.pth')
        
        writer.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], e)
        lr_scheduler.step()

    history['training_time(min)'] = total_time
    return history
