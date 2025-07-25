from config.arg_reader import read_arguments
import tempfile
from train.traineval import train_model, inference, train_experiment
from dataset.managedataset import check_and_split
from config.mode import Mode
from hyperparameter_search.sweep_wrapper import sweep_wrapper_factory
import torch
import wandb

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode, task, dev, weights_path, resnet_type, model_type = read_arguments()

    config = {
        "lr": 1e-3,
        "batch_size": 2,
        "epochs": 200,
        "data_path": "../images1000_filtered",
        "output_path": "./output",
        "allow_splitting": True,
        "box_detections_per_img": 1000,
        "train_test_split": [0.9, 0.1],  # full dataset split in train + test
        "train_val_split": [0.9, 0.1],   # train dataset split in train + val
        "crop_size": 512,
        "nms_threshold":0.3,
        "weight_decay": 1e-6,
        "patience": 10,
        # Checkpointing config
        "checkpointing": {
            "enabled": True,
            # Add as many metrics as wanted. Max if improving means increasing, Min if contrary
            "metrics": {
                "map": "max",
                "map_50": "max",
                "best_iou_per_prediction": "max",
                "best_iou_per_gt": "max"
            },
            "save_path": "checkpoints",  
            "save_last": True           
        }
    }
    sweep_config = {
            'method': 'random',
            'metric': {
                'name': 'map_50',
                'goal': 'maximize'
            },
            'parameters': {
                'batch_size': {
                    'values': [4, 6]
                },
                'lr': {
                    'distribution': 'log_uniform_values',
                    'min': 5e-6,
                    'max': 5e-3
                },
                'early_stopping_patience': {
                    'values': [5, 10, 12]
                },
                'weigth_decay': {
                    'values':  [0.0, 1e-6, 1e-4, 5e-3]
                }
            }
    }

    with tempfile.TemporaryDirectory() as tempdir:
        check_and_split(config,temp_dir=tempdir, device=device)
        
        if mode == Mode.TRAIN:
            train_model(config, tempdir, task, dev, device=device, model_type=model_type, resnet_type=resnet_type)
        elif mode == Mode.INFER:
            inference(config, tempdir, device=device, model_type=model_type, resnet_type=resnet_type, task_name=task, weights_path=weights_path)
        elif mode == Mode.EXPERIMENT:
            sweep_id = wandb.sweep(sweep_config, project="postgraduate-sat-object-detection")
            wrapper = sweep_wrapper_factory(config, sweep_config, task, dev, device, tempdir)
            wandb.agent(sweep_id, function=wrapper, count=30)

if __name__ == "__main__":
    main()