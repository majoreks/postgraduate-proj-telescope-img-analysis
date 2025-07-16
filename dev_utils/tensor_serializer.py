from collections import defaultdict
import torch


def serialize_tensor(*dicts: dict) -> dict:
    aggregated_loss_dict = defaultdict(list)

    dict = { k: v for d in dicts for k, v in d.items() }
    for k, v in dict.items():
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)

        if v.numel() == 1:
            aggregated_loss_dict[k].append(v.item())
        else:
            for i, val in enumerate(v.tolist()):
                aggregated_key = f"{k}_{i}"
                if aggregated_key not in aggregated_loss_dict:
                    aggregated_loss_dict[aggregated_key] = []
                aggregated_loss_dict[aggregated_key].append(val)
    return aggregated_loss_dict