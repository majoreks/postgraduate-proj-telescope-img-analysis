import os
import torch
import torch.nn as nn

save_path = 'output/model/model_weights.pt'

def save_model(model: nn.Module) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

def read_model(model: nn.Module, device: torch.device) -> nn.Module:
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model