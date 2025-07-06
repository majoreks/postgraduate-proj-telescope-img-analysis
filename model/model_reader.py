import os
import torch
import requests
import torch.nn as nn

save_path = 'output/model/model_weights.pt'

def save_model(model: nn.Module) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

def download_model_data() -> None:
    url = "https://www.dropbox.com/scl/fi/wi49zrkjolms4180kz9co/model_weights.pt?rlkey=t45k1r7lb5zoeju74eva0dotn&dl=1"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)

def read_model(model: nn.Module, device: torch.device, path: str | None = None) -> nn.Module:
    params = torch.load(save_path if path is None else path, map_location=device)
    model.load_state_dict(params)
    return model