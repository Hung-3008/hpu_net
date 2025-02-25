import torch
from torch import nn

def print_model_structure(model):
    """Print the structure of a PyTorch model with parameter counts."""
    print("Model Structure:")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
        print(f"\t{name}: {params:,}")
    print(f"Total Trainable Parameters: {total_params:,}")

def trace_layer_shapes(model, image_shape, label_shape, device="cuda"):
    """Create sample inputs and trace through model to debug shape issues."""
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    image = torch.randn((batch_size,) + image_shape).to(device)
    label = torch.randn((batch_size,) + label_shape).to(device)
    
    # Register hooks to print shapes
    hooks = []
    
    def hook_fn(module, input, output):
        print(f"{module.__class__.__name__} - Input shape: {[x.shape if isinstance(x, torch.Tensor) else x for x in input]}")
        if isinstance(output, torch.Tensor):
            print(f"{module.__class__.__name__} - Output shape: {output.shape}")
        else:
            print(f"{module.__class__.__name__} - Output shape: {type(output)}")
    
    # Register hooks for all modules
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(label, image)
        print("Forward pass successful!")
        print(f"Output shape: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
    except Exception as e:
        print(f"Error during forward pass: {e}")
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    model.train()
