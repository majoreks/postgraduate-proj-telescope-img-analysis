from typing import Dict, List, Tuple
import torch


def custom_collate_fn(batch: List[Tuple[torch.Tensor, List[Dict]]]) -> Tuple[torch.Tensor, List[List[Dict]]]:
    images = []
    labels = []
    for image_data, labels_data in batch:
        images.append(image_data)
        labels.append(labels_data)
    images = torch.stack(images, dim=0)
    return images, labels