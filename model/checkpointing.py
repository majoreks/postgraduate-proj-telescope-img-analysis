import os
import torch
import shutil
import copy
from datetime import datetime
from typing import Dict, Tuple

def init_checkpointing(config: dict, tempdir: str) -> Tuple[bool, str, dict, dict]:
    ckpt_cfg = config.get("checkpointing", {})
    enabled = ckpt_cfg.get("enabled", False)
    checkpoint_dir = os.path.join(tempdir, ckpt_cfg.get("save_path", "checkpoints"))

    if enabled:
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_metrics = ckpt_cfg.get("metrics", {})

    best_scores = {
        metric: float('-inf') if mode == "max" else float('inf')
        for metric, mode in checkpoint_metrics.items()
    }

    return enabled, checkpoint_dir, checkpoint_metrics, best_scores


def save_best_checkpoint(model, metric_name: str, score: float, best_scores: Dict[str, float], mode: str, checkpoint_dir: str):
    is_better = score > best_scores[metric_name] if mode == "max" else score < best_scores[metric_name]
    if is_better:
        best_scores[metric_name] = score
        ckpt_path = os.path.join(checkpoint_dir, f"best_model_{metric_name}.pt")
        torch.save(copy.deepcopy(model.state_dict()), ckpt_path)
        print(f"New best '{metric_name}' = {score:.4f} → saved to {ckpt_path}")


def save_last_checkpoint(model, checkpoint_dir: str):
    ckpt_path = os.path.join(checkpoint_dir, "last_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Last model saved to {ckpt_path}")


def persist_checkpoints(temp_checkpoint_dir: str, output_path: str, task: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_checkpoint_dir = os.path.join(output_path, "checkpoints", f"{task}_{ts}")
    os.makedirs(final_checkpoint_dir, exist_ok=True)

    for file in os.listdir(temp_checkpoint_dir):
        src = os.path.join(temp_checkpoint_dir, file)
        dst = os.path.join(final_checkpoint_dir, file)
        shutil.copy(src, dst)

    print(f"\n Saved best checkpoints to: {final_checkpoint_dir}")

def log_best_checkpoints(metric_best_epochs: dict, logger=None):
    lines = ["\n Metrics checkpoints summmary:"]
    for metric, (epoch, score) in metric_best_epochs.items():
        lines.append(f" - Metric '{metric}' improved at epoch {epoch+1} with value {score:.4f}")
    
    summary = "\n".join(lines)

    if logger is not None:
        logger.log_text(summary)
    else:
        print(summary)