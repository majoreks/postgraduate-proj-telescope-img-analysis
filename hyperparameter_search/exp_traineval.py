import torch
import albumentations as A
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from dataset.dataloader import custom_collate_fn
from dataset.telescope_dataset import TelescopeDataset
from logger.logger import Logger
from model.load_model import load_model
from model.model_reader import save_model, download_model_data, read_model
from dev_utils.plotImagesBBoxes import plotFITSImageWithBoundingBoxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.functional.detection import intersection_over_union
from model.checkpointing import init_checkpointing, save_best_checkpoint, save_last_checkpoint, persist_checkpoints, log_best_checkpoints
import os
from train.EarlyStopping import EarlyStopping
import wandb

early_stopping_metric = "map_50"


def train_single_epoch(model, images, targets, optimizer, device):
    model.train()

    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer.zero_grad()

    predictions, loss_dict = model(images, targets)
    loss = sum(loss for loss in loss_dict.values())

    loss.backward()
    optimizer.step()
    
    return predictions, loss_dict


def eval_single_epoch(model, images, device):
    model.eval()

    images = [img.to(device) for img in images]

    predictions = model(images)

    return predictions


def train_experiment(config: dict, tempdir: str, task: str, dev: bool, sweep_config: dict ,device) -> None:

    logger = Logger(task, sweep_config, dev)
    config_db = wandb.config  # ahora ya está disponible después de Logger()
    
    config_db = wandb.config

    data_transforms = A.Compose([A.RandomRotate90(p=1), A.ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True))

    train_data_path = os.path.join(config["data_path"], "train_dataset_cropped")
    joan_oro_dataset = TelescopeDataset(data_path=train_data_path, cache_dir=tempdir, transform=data_transforms, device=device)
    if dev:
        joan_oro_dataset = Subset(joan_oro_dataset, range(min(50, len(joan_oro_dataset))))
        config["epochs"] = 3
    
    torch.manual_seed(42*42)
    train_dataset, val_dataset = torch.utils.data.random_split(joan_oro_dataset, config['train_val_split'])

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=config_db.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config_db.batch_size, collate_fn=custom_collate_fn)

    logger = Logger(task, config, dev)
    model = load_model(device, config, config_db.nms_threshold)

    model = model.train()
    optimizer = torch.optim.Adam(list(model.backbone.parameters()) + list(model.roi_heads.box_predictor.parameters()), lr=config_db.learning_rate, weight_decay=config_db.weight_decay)
    logger.log_model(model)

    start = 0.3 * config["box_detections_per_img"]
    end = config["box_detections_per_img"]
    detection_thresholds = [
        int(start),
        int((start + end) / 2),
        int(end)
    ]
    mAPMetric = MeanAveragePrecision(iou_type="bbox", max_detection_thresholds=detection_thresholds, backend='faster_coco_eval')

    # Checkpointing Setup
    checkpoint_enabled, checkpoint_dir, checkpoint_metrics, best_scores = init_checkpointing(config, tempdir)
    save_last = config.get("checkpointing", {}).get("save_last", True)
    metric_best_epochs = {}

    early_stopping = EarlyStopping(early_stopping_metric, patience=config_db.patience)

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        train_losses = {}  # will hold lists of each loss term

        for _, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"train | epoch {epoch+1}"):
            if len(images) == len(targets) == 0:
                print("Batch size 0, skipping")
                continue

            predictions, loss_dict = train_single_epoch(model, images, targets, optimizer, device)
            for k, v in loss_dict.items():
                train_losses.setdefault(k, []).append(v.item())

        avg_train = {k: torch.tensor(sum(vals)/len(vals), dtype=torch.float32) for k, vals in train_losses.items()}
        logger.log_train_loss(avg_train, is_train=True)

        mAPMetric.reset()
        ious_dim0 = []
        ious_dim1 = []

        with torch.no_grad():
            for _, (images, targets) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"eval | epoch {epoch+1}"):
                predictions = eval_single_epoch(model, images, device=device)

                predictions = predictions[0]
                predictions = [{k: v.detach().cpu() for k, v in pred.items()} for pred in predictions]

                mAPMetric.update(predictions, targets)
                for pred, target in zip(predictions, targets):
                    iou = intersection_over_union(pred["boxes"], target["boxes"], aggregate=False)
                    ious_dim0.append(iou.max(dim=0).values)
                    ious_dim1.append(iou.max(dim=1).values)


        mAPMetrics = mAPMetric.compute()
        mAPMetrics.pop("classes", None)

        iou_dim0 = torch.cat(ious_dim0).mean()       # 1-D tensor of length = sum of all element-counts
        iou_dim1 = torch.cat(ious_dim1).mean()

        iou_metrics = {
            "best_iou_per_gt": iou_dim0, "best_iou_per_prediction": iou_dim1
        }
        logger.log_train_loss(mAPMetrics, iou_metrics, is_train=False)
        logger.step()

        all_metrics = {**mAPMetrics, **iou_metrics}

        early_stopping.step(all_metrics)
        if early_stopping.should_stop:
            print(f"Early stopping: no improvement in [{early_stopping_metric}]")
            logger.log_early_stop()
            break

        # OOOOJOOOO: every time a metric is uploaded, modifications to all_metrics is needed. We should improve the config and
        # treat all the metrics in the same way (metrics.py)
        if checkpoint_enabled:
            for metric_name, mode in checkpoint_metrics.items():
                score = all_metrics.get(metric_name)

                if score is None:
                    print(f"Metric '{metric_name}' not found.")
                    continue
                print(f"{metric_name} : {score} and {best_scores[metric_name]}")
                is_better = score > best_scores[metric_name] if mode == "max" else score < best_scores[metric_name]
                if is_better:
                    save_best_checkpoint(model, metric_name, score, best_scores, mode, checkpoint_dir)
                    best_scores[metric_name] = score
                    metric_best_epochs[metric_name] = (epoch, score)

            if save_last:
                save_last_checkpoint(model, checkpoint_dir)

    save_model(model)

    # Copy the best models to a persistent folder from the tempdir
    if checkpoint_enabled:
        temp_checkpoint_dir = os.path.join(tempdir, config["checkpointing"]["save_path"])
        persist_checkpoints(temp_checkpoint_dir, config["output_path"], task)

        # [MODIFICADO] Log resumen final de checkpoints
        log_best_checkpoints(metric_best_epochs, logger)
    
    logger.flush()