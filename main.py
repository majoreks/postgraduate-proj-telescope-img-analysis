import torch
import tempfile
import os
import albumentations as A
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from config.arg_reader import read_arguments
from config.mode import Mode
from dataset.dataloader import custom_collate_fn
from dataset.telescope_dataset import TelescopeDataset
from config.device import get_device
from logger.logger import Logger
from model.load_model import load_model
from model.model_reader import save_model, download_model_data, read_model
from dev_utils.plotImagesBBoxes import plotFITSImageWithBoundingBoxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def eval_single_epoch(model, images):
    model.eval()

    images = [img.to(device) for img in images]

    predictions = model(images)

    return predictions


def train_model(config: dict, tempdir: str, task: str, dev: bool) -> None:

    data_transforms = A.Compose([A.AtLeastOneBBoxRandomCrop(width=512, height=512), A.RandomRotate90(p=1), A.ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True))

    train_data_path = os.path.join(config["data_path"], "train_dataset")
    joan_oro_dataset = TelescopeDataset(data_path=train_data_path, cache_dir=tempdir, transform=data_transforms, device=device)
    if dev:
        joan_oro_dataset = Subset(joan_oro_dataset, range(min(50, len(joan_oro_dataset))))
        config["epochs"] = 5
    
    torch.manual_seed(42*42)
    train_dataset, val_dataset = torch.utils.data.random_split(joan_oro_dataset, [0.82, 0.18])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=custom_collate_fn) # TODO add validation

    logger = Logger(task, config, dev)

    model = load_model(device)

    model = model.train()
    optimizer = torch.optim.Adam(list(model.backbone.parameters()) + list(model.roi_heads.box_predictor.parameters()), lr=1e-4, weight_decay=0.001)

    logger.log_model(model)

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        for _, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"train | epoch {epoch+1}"):
            if len(images) == len(targets) == 0:
                print("Batch size 0, skipping")
                continue

            predictions, loss_dict = train_single_epoch(model, images, targets, optimizer, device)

            logger.log_train_loss(loss_dict, True)

        metric = MeanAveragePrecision(iou_type="bbox")
        with torch.no_grad():
            for _, (images, targets) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"eval | epoch {epoch+1}"):
                predictions = eval_single_epoch(model, images)
                predictions = predictions[0]
                predictions = [{k: v.detach().cpu() for k, v in pred.items()} for pred in predictions]
                metric.update(predictions, targets)

                logger.log_train_loss(metric.compute(), False)

    save_model(model)
    logger.flush()


def inference(config, tempdir):
    '''print('inference',path)
    print('-----')
    test_data_path = os.path.join(config["data_path"], "test_dataset")
    joan_oro_dataset = TelescopeDataset(data_path=test_data_path, cache_dir=tempdir, transform=data_transforms, device=device)

    #dataset = TelescopeDataset(data_path=path, cache_dir=tempdir, transform=data_transforms, device=device)
    #joan_oro_dataset = TelescopeDataset(data_path=config["data_path"], cache_dir=tempdir,
                                        transform=data_transforms, device=device)
    data_transforms = A.Compose([A.ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True))
    test_loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)

    model = load_model(device, True)

    #print(len(test_loader))
    #print(model)
    print(f"Número de muestras cargadas: {len(dataset)}")'''

    test_data_path = os.path.join(config["data_path"], "test_dataset")

    print('inference', test_data_path)
    print('-----')
    print("📁 Directorio de datos:", test_data_path)
    if not os.path.exists(test_data_path):
        print("❌ ERROR: El directorio no existe.")
    else:
        print("✅ El directorio existe.")
        # print("📦 Archivos encontrados:", os.listdir(test_data_path))


    # data_transforms = A.Compose(
    #     [A.ToTensorV2()],
    #     bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True)
    # )
    
    data_transforms = A.Compose([A.AtLeastOneBBoxRandomCrop(width=512, height=512), A.ToTensorV2()], 
                                 bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], 
                                                          filter_invalid_bboxes=True))


    dataset = TelescopeDataset(data_path=test_data_path, cache_dir=tempdir, transform=data_transforms, device=device)
    test_loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)

    download_model_data()
    model = load_model(device, load_weights =True)
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

            # Almacenar resultados
            results.append({
                'image_index': idx,
                'image_tensor': images[0].cpu(),
                'ground_truth': targets[0],   # dict con 'boxes' y 'labels'
                'prediction': predictions[0]  # dict con 'boxes', 'labels', 'scores'
            })

            print(f"[{idx}] GT: {len(targets[0]['boxes'])} BBs, Pred: {len(predictions[0]['boxes'])} BBs")

    # Opcional: guardar en disco como torch.save
    output_path = os.path.join(tempdir, "results.pt")
    torch.save(results, output_path)
    print(f"\nResultados guardados en: {output_path}")

    for i in range(max(3, len(results))):
        image = results[i]['image_tensor'].cpu().numpy()
        predictions = results[i]['prediction']  # o 'ground_truth' si quieres comparar
        ground_truth = results[i]['ground_truth']

        plotFITSImageWithBoundingBoxes(image, labels_predictions=predictions, labels_ground_truth=ground_truth, save_fig=True, save_fig_sufix=str(results[i]['image_index']))

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
def split_dataset(config: dict, tempdir: str) -> None:

    data_path = config["data_path"]

    gz_files = []
    for root, _, files in os.walk(data_path):
        gz_files.extend([os.path.join(root, f) for f in files if f.endswith('.gz')])
    dat_files = []
    for root, _, files in os.walk(data_path):
        dat_files.extend([os.path.join(root, f) for f in files if f.endswith('.dat')])

    for file_path in gz_files:
        dest_path = os.path.join(data_path, os.path.basename(file_path))
        if file_path != dest_path:
            os.rename(file_path, dest_path)

    for file_path in dat_files:
        dest_path = os.path.join(data_path, os.path.basename(file_path))
        if file_path != dest_path:
            os.rename(file_path, dest_path)

    # Remove all empty folders recursively under data_path
    for root, dirs, files in os.walk(data_path, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

    joan_oro_dataset = TelescopeDataset(data_path=data_path, cache_dir=tempdir, transform=None, device=device)
    joan_oro_dataset.move_empty_to_folder(os.path.join(config['data_path'], "metadataless_dataset"))
        
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(joan_oro_dataset, [0.82, 0.18])
    train_dataset.dataset.move_dataset_to_folder(os.path.join(config['data_path'], "train_dataset"), indexes=train_dataset.indices)
    test_dataset.dataset.move_dataset_to_folder(os.path.join(config['data_path'], "test_dataset"), indexes=test_dataset.indices)

    

def main() -> None:
    mode, task, dev = read_arguments()

    config = {
        "lr": 1e-3,
        "batch_size": 2,
        "epochs": 15,
        "data_path": "/home/szymon/data/posgrado-proj/images1000",
        "allow_splitting": True
    }

    with tempfile.TemporaryDirectory() as tempdir:
        
        if config["allow_splitting"] == True:
            data_path = config["data_path"]
            print("Listing folders in:", data_path)
            need_to_split = True
            if os.path.exists(data_path):
                if len(os.listdir(data_path)) == 3:
                    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(config["data_path"], f))]
                    if set(folders) == {"metadataless_dataset", "test_dataset", "train_dataset"}:
                        need_to_split = False
                        print("Dataset already split into train and test folders.")
            
            if need_to_split == True:
                print("Dataset not split, merging all files and splitting now...")
                split_dataset(config, tempdir)

        if mode == Mode.TRAIN:
            train_model(config, tempdir, task, dev)
        elif mode == Mode.INFER:
            inference(config, tempdir)

if __name__ == "__main__":
    main()
