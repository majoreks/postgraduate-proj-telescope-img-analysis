from config.arg_reader import read_arguments
import tempfile
import os
from traineval import train_model, inference
from dataset.managedataset import check_and_split
from config.mode import Mode
from dataset.dataloader import custom_collate_fn
from dataset.telescope_dataset import TelescopeDataset
from config.device import get_device
# from model.load_model import load_model
# from model.model_reader import save_model
from postprocess.plot_losses import plot_losses

import torch


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode, task, dev = read_arguments()

    config = {
        "lr": 1e-3,
        "batch_size": 2,
        "epochs": 15,
        "data_path": "../images1000",
        "allow_splitting": True,
        "box_detections_per_img": 1000,
        "train_test_split": [0.9, 0.1], # full dataset split in train + test
        "train_val_split": [0.9, 0.1], # train dataset split in train + val
        "crop_size": 512,
        'nms_threshold': 0.4
    }
    mode = Mode.TRAIN


    with tempfile.TemporaryDirectory() as tempdir:
    
        check_and_split(config,temp_dir=tempdir, device=device)
        
        if mode == Mode.TRAIN:
            train_model(config, tempdir, task, dev, device=device)
        elif mode == Mode.INFER:
            inference(config, tempdir, device=device)

if __name__ == "__main__":
    main()
