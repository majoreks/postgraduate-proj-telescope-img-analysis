from typing import Dict, List, Tuple
import torch


def custom_collate_fn(batch: List[Tuple[torch.Tensor, List[Dict]]]) -> Tuple[torch.Tensor, List[List[Dict]]]:
    images = []
    labels = []

    def filter_batch(sample):
        _, y = sample
        return len(y["labels"]) != 0

    filtered_batch = [sample for sample in batch if filter_batch(sample)]
    if len(filtered_batch) == 0:
        return images, labels

    for image_data, labels_data in filtered_batch:
        images.append(image_data)
        labels.append(labels_data)
    images = torch.stack(images, dim=0)
    return images, labels