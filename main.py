import torch
import tempfile
import os
import albumentations as A
from torch.utils.data import DataLoader
from tqdm import tqdm
from config.arg_reader import read_arguments
from config.mode import Mode
from dataset.dataloader import custom_collate_fn
from dataset.telescope_dataset import TelescopeDataset
from config.device import get_device
from logger.logger import Logger
from model.load_model import load_model
from model.model_reader import save_model
from dev_utils.plotImagesBBoxes import plotFITSImageWithBoundingBoxes

device = get_device()

def train_single_epoch(model, images, targets, optimizer, device):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer.zero_grad()

    predictions, loss_dict = model(images, targets)
    loss = sum(loss for loss in loss_dict.values())

    loss.backward()
    optimizer.step()
    
    return predictions, loss_dict


def eval_single_epoch(model, images, targets):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    predictions, loss_dict = model(images, targets)

    return predictions, loss_dict


def train_model(config: dict, tempdir: str, task: str, dev: bool) -> None:

    data_transforms = A.Compose([A.AtLeastOneBBoxRandomCrop(width=512, height=512), A.RandomRotate90(p=1), A.ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True))

    joan_oro_dataset = TelescopeDataset(data_path=config["train_data_path"], cache_dir=tempdir, transform=data_transforms, device=device)

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
            predictions, loss_dict = train_single_epoch(model, images, targets, optimizer, device)

            logger.log_train_loss(loss_dict, True)

        with torch.no_grad():
            for _, (images, targets) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"eval | epoch {epoch+1}"):
                predictions, loss_dict = eval_single_epoch(model, images, targets)

                logger.log_train_loss(loss_dict, False)

    save_model(model)
    logger.flush()


def inference(path, tempdir):
    '''print('inference',path)
    print('-----')
    #dataset = TelescopeDataset(data_path=path, cache_dir=tempdir, transform=data_transforms, device=device)
    #joan_oro_dataset = TelescopeDataset(data_path=config["train_data_path"], cache_dir=tempdir,
                                        transform=data_transforms, device=device)
    data_transforms = A.Compose([A.ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True))


    train_dataset, val_dataset = torch.utils.data.random_split(joan_oro_dataset, [0.82, 0.18])
    test_loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)

    model = load_model(device, True)

    #print(len(test_loader))
    #print(model)
    print(f"Número de muestras cargadas: {len(dataset)}")'''

    print('inference', path)
    print('-----')
    print("📁 Directorio de datos:", path)
    if not os.path.exists(path):
        print("❌ ERROR: El directorio no existe.")
    else:
        print("✅ El directorio existe.")
        print("📦 Archivos encontrados:", os.listdir(path))


    data_transforms = A.Compose(
        [A.ToTensorV2()],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True)
    )

    dataset = TelescopeDataset(data_path=path, cache_dir=tempdir, transform=data_transforms, device=device)
    test_loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)

    model = load_model(device, load_weights =True)
    model.eval()

    print(f"Número de muestras cargadas: {len(dataset)}")

    results = []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_loader):
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

    for i in range(min(3, len(results))):
        image = results[i]['image_tensor'].cpu().numpy()
        labels = results[i]['prediction']  # o 'ground_truth' si quieres comparar

        plotFITSImageWithBoundingBoxes(image, labels, save_fig=False)

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

def main() -> None:
    mode, task, dev = read_arguments()

    config = {
        "lr": 1e-3,
        "batch_size": 6,
        "epochs": 15,
        "train_data_path": "data_full"
    }

    with tempfile.TemporaryDirectory() as tempdir:
        if mode == Mode.TRAIN:
            train_model(config, tempdir, task, dev)

        elif mode == Mode.INFER:
            inference('data_inference', tempdir)

if __name__ == "__main__":
    main()
