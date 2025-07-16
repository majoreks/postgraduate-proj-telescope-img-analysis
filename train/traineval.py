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


def train_model(config: dict, tempdir: str, task: str, dev: bool, device) -> None:

    data_transforms = A.Compose([A.RandomRotate90(p=1), A.ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True))

    train_data_path = os.path.join(config["data_path"], "train_dataset_cropped")
    joan_oro_dataset = TelescopeDataset(data_path=train_data_path, cache_dir=tempdir, transform=data_transforms, device=device)
    if dev:
        joan_oro_dataset = Subset(joan_oro_dataset, range(min(50, len(joan_oro_dataset))))
        config["epochs"] = 3
    
    torch.manual_seed(42*42)
    train_dataset, val_dataset = torch.utils.data.random_split(joan_oro_dataset, config['train_val_split'])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=custom_collate_fn)

    logger = Logger(task, config, dev)
    model = load_model_v2(device, config)

    model = model.train()
    optimizer = torch.optim.Adam(list(model.backbone.parameters()) + list(model.roi_heads.box_predictor.parameters()), lr=config["lr"], weight_decay=config["weight_decay"])
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

    early_stopping = EarlyStopping(early_stopping_metric, patience=config["patience"])

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

def train_experiment(config: dict, tempdir: str, task: str, dev: bool, device, sweep_params, on_batch_end=None) -> None:
    logger = Logger(task, sweep_params, dev)

    data_transforms = A.Compose([
        A.RandomRotate90(p=1),
        A.ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True))

    train_data_path = os.path.join(config["data_path"], "train_dataset_cropped")
    dataset = TelescopeDataset(train_data_path, cache_dir=tempdir, transform=data_transforms, device=device)

    print(f"[DEBUG] Dataset root path: {train_data_path}")
    print(f"[DEBUG] Nº de muestras cargadas: {len(dataset)}")

    if dev:
        dataset = Subset(dataset, range(min(50, len(dataset))))
        config["epochs"] = 3

    torch.manual_seed(42 * 42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, config["train_val_split"])

    # Parámetros
    batch_size = sweep_params.get("batch_size", 4)
    lr = sweep_params.get("lr", 1e-3)
    weight_decay = sweep_params.get("weight_decay", 1e-4)
    patience = sweep_params.get("early_stopping_patience", 0)

    print(f"""
        Sweep Parameters:
        - batch_size              = {batch_size}
        - learning_rate (lr)      = {lr}
        - weight_decay            = {weight_decay}
        - early_stopping_patience = {patience}
    """)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

    model = load_model_v2(device, config)

    model = model.train()
    optimizer = torch.optim.Adam(
        list(model.backbone.parameters()) + list(model.roi_heads.box_predictor.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    logger.log_model(model)

    # Métricas
    start = 0.3 * config["box_detections_per_img"]
    end = config["box_detections_per_img"]
    detection_thresholds = [int(start), int((start + end) / 2), int(end)]
    mAPMetric = MeanAveragePrecision(iou_type="bbox", max_detection_thresholds=detection_thresholds, backend='faster_coco_eval')

    # Checkpointing
    checkpoint_enabled, checkpoint_dir, checkpoint_metrics, best_scores = init_checkpointing(config, tempdir)
    save_last = config.get("checkpointing", {}).get("save_last", True)
    metric_best_epochs = {}

    early_stopping_metric = config.get("checkpointing", {}).get("early_stopping_metric", "map_50")
    early_stopping = EarlyStopping(early_stopping_metric, patience=patience)
    epochs = config['epochs']
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_losses = {}

        for _, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"train | epoch {epoch+1}"):
            if len(images) == 0 or len(targets) == 0:
                print("Batch vacío, saltando")
                continue

            predictions, loss_dict = train_single_epoch(model, images, targets, optimizer, device)
            for k, v in loss_dict.items():
                train_losses.setdefault(k, []).append(v.item())
            if on_batch_end is not None:
                on_batch_end() 

        avg_train = {k: sum(v)/len(v) for k, v in train_losses.items()}
        logger.log_train_loss(avg_train, is_train=True)

        mAPMetric.reset()
        ious_dim0, ious_dim1 = [], []

        with torch.no_grad():
            for _, (images, targets) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"eval | epoch {epoch+1}"):
                predictions = eval_single_epoch(model, images, device)
                predictions = [{k: v.detach().cpu() for k, v in pred.items()} for pred in predictions[0]]
                mAPMetric.update(predictions, targets)

                for pred, target in zip(predictions, targets):
                    iou = intersection_over_union(pred["boxes"], target["boxes"], aggregate=False)
                    ious_dim0.append(iou.max(dim=0).values)
                    ious_dim1.append(iou.max(dim=1).values)

        mAPMetrics = mAPMetric.compute()
        mAPMetrics.pop("classes", None)

        iou_dim0 = torch.cat(ious_dim0).mean()
        iou_dim1 = torch.cat(ious_dim1).mean()
        iou_metrics = {
            "best_iou_per_gt": iou_dim0,
            "best_iou_per_prediction": iou_dim1
        }

        logger.log_train_loss(mAPMetrics, iou_metrics, is_train=False)
        logger.step()

        all_metrics = {**mAPMetrics, **iou_metrics}
        early_stopping.step(all_metrics)

        if early_stopping.should_stop:
            print(f"Early stopping: no improvement in [{early_stopping_metric}]")
            logger.log_early_stop()
            break

        if checkpoint_enabled:
            for metric_name, mode in checkpoint_metrics.items():
                score = all_metrics.get(metric_name)
                if score is None:
                    print(f"Metric '{metric_name}' not found.")
                    continue

                is_better = score > best_scores[metric_name] if mode == "max" else score < best_scores[metric_name]
                if is_better:
                    save_best_checkpoint(model, metric_name, score, best_scores, mode, checkpoint_dir)
                    best_scores[metric_name] = score
                    metric_best_epochs[metric_name] = (epoch, score)

            if save_last:
                save_last_checkpoint(model, checkpoint_dir)

    save_model(model)

    if checkpoint_enabled:
        persist_checkpoints(os.path.join(tempdir, config["checkpointing"]["save_path"]), config["output_path"], task)
        log_best_checkpoints(metric_best_epochs, logger)

    logger.flush()