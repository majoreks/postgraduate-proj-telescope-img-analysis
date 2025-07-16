import torch.nn as nn

def count_parameters(model: nn.Module) -> int:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    non_trainable = total - trainable
    
    return format_params(total), format_params(trainable), format_params(non_trainable)

def format_params(n: int) -> str:
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(n)