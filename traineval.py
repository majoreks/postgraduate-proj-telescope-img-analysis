import torch
import copy
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
from torchmetrics.detection.iou import IntersectionOverUnion
import os


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


def train_model(config: dict, tempdir: str, task: str, dev: bool, device) -> None:

    data_transforms = A.Compose([A.AtLeastOneBBoxRandomCrop(width=config["crop_size"], height=config["crop_size"]), A.RandomRotate90(p=1), A.ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True))

    train_data_path = os.path.join(config["data_path"], "train_dataset")
    joan_oro_dataset = TelescopeDataset(data_path=train_data_path, cache_dir=tempdir, transform=data_transforms, device=device)
    if dev:
        joan_oro_dataset = Subset(joan_oro_dataset, range(min(50, len(joan_oro_dataset))))
        config["epochs"] = 3
    
    torch.manual_seed(42*42)
    train_dataset, val_dataset = torch.utils.data.random_split(joan_oro_dataset, config['train_val_split'])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=custom_collate_fn) # TODO add validation

    logger = Logger(task, config, dev)
    model = load_model(device, config)

    model = model.train()
    optimizer = torch.optim.Adam(list(model.backbone.parameters()) + list(model.roi_heads.box_predictor.parameters()), lr=1e-4, weight_decay=0.001)

    logger.log_model(model)

    mAPMetric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    mAPMetric.warn_on_many_detections = False # https://stackoverflow.com/a/76957869 we have possibly more than 100 detections, metric calculation takes into account first n (by score) detections 
    iou_metric = IntersectionOverUnion(
        class_metrics=True
    )

    # --- Checkpointing Setup ---
    ckpt_cfg = config.get("checkpointing", {})
    checkpoint_enabled = ckpt_cfg.get("enabled", False)
    save_last = ckpt_cfg.get("save_last", True)
    checkpoint_dir = os.path.join(tempdir, ckpt_cfg.get("save_path", "checkpoints"))
    if checkpoint_enabled:
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_metrics = ckpt_cfg.get("metrics", {})
    best_scores = {
        metric: float('-inf') if mode == "max" else float('inf')
        for metric, mode in checkpoint_metrics.items()
    }

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
        iou_metric.reset()

        with torch.no_grad():
            for _, (images, targets) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"eval | epoch {epoch+1}"):
                predictions = eval_single_epoch(model, images, device=device)

                predictions = predictions[0]
                predictions = [{k: v.detach().cpu() for k, v in pred.items()} for pred in predictions]

                mAPMetric.update(predictions, targets)
                iou_metric.update(predictions, targets)

        mAPMetrics = mAPMetric.compute()
        iouMetrics = iou_metric.compute()
        mAPMetrics.pop("classes", None)

        logger.log_train_loss(mAPMetrics, iouMetrics, is_train=False)
        logger.step()

        # --- Checkpointing per metric ---
        if checkpoint_enabled:
            for metric_name, mode in checkpoint_metrics.items():
                score = mAPMetrics.get(metric_name)
                if score is None:
                    print(f"⚠️ Métrica '{metric_name}' no encontrada.")
                    continue

                is_better = score > best_scores[metric_name] if mode == "max" else score < best_scores[metric_name]
                if is_better:
                    best_scores[metric_name] = score
                    ckpt_path = os.path.join(checkpoint_dir, f"best_model_{metric_name}.pt")
                    torch.save(copy.deepcopy(model.state_dict()), ckpt_path)
                    print(f"New best '{metric_name}' = {score:.4f} → checkpoint saved in {ckpt_path}")

            # --- Save last model if required ---
        if checkpoint_enabled and save_last:
            last_ckpt_path = os.path.join(checkpoint_dir, "last_model.pt")
            torch.save(model.state_dict(), last_ckpt_path)
            print(f"Last model saved in {last_ckpt_path}")


    save_model(model)
    logger.flush()


def inference(config, tempdir, device, save_fig=True):
    
    test_data_path = os.path.join(config["data_path"], "test_dataset")

    print('inference', test_data_path)
    print('-----')
    print("Directorio de datos:", test_data_path)
    if not os.path.exists(test_data_path):
        print("ERROR: El directorio no existe.")
    else:
        print("El directorio existe.")

    
    data_transforms = A.Compose([A.AtLeastOneBBoxRandomCrop(width=config["crop_size"], height=config["crop_size"]), A.ToTensorV2()], 
                                 bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], 
                                                          filter_invalid_bboxes=True))


    dataset = TelescopeDataset(data_path=test_data_path, cache_dir=tempdir, transform=data_transforms, device=device)
    test_loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)

    download_model_data()
    model = load_model(device, config=config, load_weights=True)
    model = read_model(model, device)
    model.eval()

    print(f"Número de muestras cargadas: {len(dataset)}")

    results = []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_loader):
            if not images:
                print(f"Skipping idx {idx} because images is empty.")
                continue
            
            if not targets:
                print(f"Skipping idx {idx} because targets is empty.")
                continue

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Obtener predicciones del modelo
            predictions, _ = model(images, targets)

            # Aux variables
            image=images[0].cpu().numpy()
            ground_truth=targets[0]
            prediction=predictions[0]
            filename=dataset.images_list[idx].split(".fits")[0].replace(".", "_")

            # Print results from the model
            print(f"[{idx}] Image {filename} GT: {len(targets[0]['boxes'])} BBs, Pred: {len(predictions[0]['boxes'])} BBs")

            # Saves the inference plotted image
            if save_fig:
                plotFITSImageWithBoundingBoxes(image, labels_predictions=prediction, labels_ground_truth=ground_truth, save_fig=True, save_fig_sufix=f"{idx}_{filename}", title_sufix=filename)              

            # Almacenar resultados
            results.append({
                'image_index': idx,
                'image_tensor': images[0].cpu().numpy(),
                'ground_truth': ground_truth,   # dict con 'boxes' y 'labels'
                'prediction': prediction,  # dict con 'boxes', 'labels', 'scores'
                'filename': filename  # nombre del archivo de imagen
            })

    # Opcional: guardar en disco como torch.save
    output_path = os.path.join(tempdir, "results.pt")
    torch.save(results, output_path)
    print(f"\nResultados guardados en: {output_path}")

    #avaluar el model sobre les dades de test:
        #1. filtrar objecte dataset a la regió test del dataset (szimon to harmonize)
        #2. Iterar la regió del dataset i avaluar el model sobre cada element de la iteració
                #com es guarda això
        #3. guardar la sortida del model (optimitzant el recàlcul si es pot - prog. dinàmica/fer cachés)
        #4. Processar la sortida i transformat la sortida (matriu) en imatge
        #5. Pintar les BB model i BB GT per cada imatge
        #6. mètriques... [distància de la BB M i la BB GT]

    #estructura de dades com s'itera
    #utilitzar aquelles que hagin anat a parar al badge de test