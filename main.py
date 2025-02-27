import argparse
import datetime
import json
import time
import socket
import tracemalloc
import os
import yaml

import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

from dataset_ACDC import *
from model import *   # Make sure your HPUNet and other modules are imported from here
from train import *
from torch.utils.tensorboard import SummaryWriter

# Define a CrossEntropyLossWrapper for segmentation
class CrossEntropyLossWrapper(torch.nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__(reduction='none')
        self.last_loss = None

    def forward(self, input, target, **kwargs):
        # input shape: (batch, classes, H, W)
        # target shape: (batch, H, W) with class indices
        loss = super().forward(input, target)
        self.last_loss = {'expanded_loss': loss}
        return loss

def load_config(config_path):
    """
    Load configuration from YAML file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def config_to_args(config):
    """
    Convert configuration dict to argparse namespace for compatibility
    """
    args = argparse.Namespace()
    
    # Flatten config into args
    for section in config:
        for key, value in config[section].items():
            setattr(args, key, value)
    
    return args

# Command line argument to specify config file
parser = argparse.ArgumentParser(description="Hierarchical Probabilistic U-Net for ACDC")
parser.add_argument("--config", default="config.yml", help="Path to config file")
parser.add_argument("--override", nargs="+", default=[], help="Override config parameters, format: key=value")
cmd_args = parser.parse_args()

# Load config and convert to args
config = load_config(cmd_args.config)

# Override config with command line arguments if provided
for override in cmd_args.override:
    if '=' in override:
        key, value = override.split('=', 1)
        # Find the section this key belongs to
        for section in config:
            if key in config[section]:
                # Try to convert value to the same type as in config
                orig_type = type(config[section][key])
                if orig_type == bool:
                    config[section][key] = value.lower() in ('true', '1', 't')
                elif orig_type == list:
                    # Handle lists based on their element type
                    if config[section][key] and isinstance(config[section][key][0], int):
                        config[section][key] = [int(x) for x in value.split(',')]
                    elif config[section][key] and isinstance(config[section][key][0], float):
                        config[section][key] = [float(x) for x in value.split(',')]
                    else:
                        config[section][key] = value.split(',')
                elif orig_type == int:
                    config[section][key] = int(value)
                elif orig_type == float:
                    config[section][key] = float(value)
                else:
                    config[section][key] = value
                break

# Convert config to args
args = config_to_args(config)

# Adjust Arguments
if args.latent_locks is None:
    args.latent_locks = [0] * args.latent_num
args.latent_locks = [bool(l) for l in args.latent_locks]

if len(args.kernel_size) < len(args.intermediate_ch):
    if len(args.kernel_size) == 1:
        args.kernel_size = args.kernel_size * len(args.intermediate_ch)
    else:
        print('Invalid kernel size, exiting...')
        exit()

if len(args.dilation) < len(args.intermediate_ch):
    if len(args.dilation) == 1:
        args.dilation = args.dilation * len(args.intermediate_ch)
    else:
        print('Invalid dilation, exiting...')
        exit()

if len(args.scale_depth) < len(args.intermediate_ch):
    if len(args.scale_depth) == 1:
        args.scale_depth = args.scale_depth * len(args.intermediate_ch)
    else:
        print('Invalid scale depth, exiting...')
        exit()

# Set Random Seed
if args.random_seed is not None:
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
else:
    np.random.seed(0)
    torch.manual_seed(0)

# Set Device
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.available_gpus = torch.cuda.device_count()
device = torch.device(args.device)
print("Device is {}".format(device))

# Generate Stamp
timestamp = datetime.datetime.now().strftime('%m%d-%H%M')
compute_node = socket.gethostname()
suffix = datetime.datetime.now().strftime('%f')
stamp = timestamp + '_' + compute_node[:2] + '_' + suffix + '_' + args.comment
print('Stamp:', stamp)
args.compute_node = compute_node
args.stamp = stamp

# Create output directory if it doesn't exist
os.makedirs(f'{args.output_dir}/{stamp}', exist_ok=True)

# Initialize SummaryWriter (for tensorboard)
writer = SummaryWriter('{}/{}/tb'.format(args.output_dir, stamp))

# Load Data
train_data = ACDCdataset(base_dir='datasets/ACDC', list_dir='datasets/ACDC/lists_ACDC', split='train')
train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4)

sample_batch = next(iter(train_loader))
sample_image = sample_batch['image'][0]  # Get first image from batch
_, height, width = sample_image.shape if len(sample_image.shape) == 3 else (1, sample_image.shape[0], sample_image.shape[1])
args.pixels = height * width

val_data = ACDCdataset(base_dir='datasets/ACDC', list_dir='datasets/ACDC/lists_ACDC', split='valid')
val_loader = DataLoader(val_data, batch_size=2, shuffle=True, num_workers=4)

# Initialize Model
model = HPUNet(in_ch=args.in_ch, out_ch=args.out_ch, chs=args.intermediate_ch,
               latent_num=args.latent_num, latent_channels=args.latent_chs, latent_locks=args.latent_locks,
               scale_depth=args.scale_depth, kernel_size=args.kernel_size, dilation=args.dilation,
               padding_mode=args.padding_mode).float()

args.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model.to(device)

# Set Loss Function
## Reconstruction Loss
if args.rec_type.lower() == 'mse':
    reconstruction_loss = MSELossWrapper()
elif args.rec_type.lower() == 'crossentropy':
    reconstruction_loss = CrossEntropyLossWrapper()
else:
    print('Invalid reconstruction loss type, exiting...')
    exit()

## Total Loss
if args.loss_type.lower() == 'elbo':
    if args.beta_asc_steps is None:
        beta_scheduler = BetaConstant(args.beta)
    else:
        beta_scheduler = BetaLinearScheduler(ascending_steps=args.beta_asc_steps, constant_steps=args.beta_cons_steps,
                                               max_beta=args.beta, saturation_step=args.beta_saturation_step)
    criterion = ELBOLoss(reconstruction_loss=reconstruction_loss, beta=beta_scheduler).to(device)
elif args.loss_type.lower() == 'geco':
    kappa = args.kappa
    if args.kappa_px is True:
        kappa *= args.pixels
    criterion = GECOLoss(reconstruction_loss=reconstruction_loss, kappa=kappa, decay=args.decay,
                          update_rate=args.update_rate, device=device).to(device)
else:
    print('Invalid loss type, exiting...')
    exit()

# Set Optimizer
if args.optimizer == 'adamax':
    optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.optimizer == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    print('Optimizer not known, exiting...')
    exit()

# Set LR Scheduler
if args.scheduler_type == 'cons':
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs)
elif args.scheduler_type == 'step':
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
elif args.scheduler_type == 'milestones':
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)

# Save Args and Config
argsdict = vars(args)
with open('{}/{}/args.json'.format(args.output_dir, stamp), 'w') as f:
    json.dump(argsdict, f)
with open('{}/{}/args.txt'.format(args.output_dir, stamp), 'w') as f:
    for k in argsdict.keys():
        f.write("'{}': '{}'\n".format(k, argsdict[k]))
        
# Also save the original config used
with open('{}/{}/config.yml'.format(args.output_dir, stamp), 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

# Start Timing
start = time.time()

# Train the Model
history = train_model(args, model, train_loader, criterion, optimizer, lr_scheduler, writer, device, val_loader, start)

# End Timing & Report Training Time
end = time.time()
training_time = (end - start) / 3600
history['training_time(hours)'] = training_time
print('Training done in {:.1f} hours'.format(training_time))

# Save Model, Loss and History
torch.save(model, '{}/{}/model.pth'.format(args.output_dir, stamp))
torch.save(criterion, '{}/{}/loss.pth'.format(args.output_dir, stamp))
with open('{}/{}/history.json'.format(args.output_dir, stamp), 'w') as f:
    json.dump(history, f)