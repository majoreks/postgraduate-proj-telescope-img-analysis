import torch
import tempfile
import os
import albumentations as A
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from config.arg_reader import read_arguments
from config.mode import Mode
from dataset.dataloader import custom_collate_fn
from dataset.telescope_dataset import TelescopeDataset
from config.device import get_device
# from model.load_model import load_model
# from model.model_reader import save_model
from postprocess.plot_losses import plot_losses
from dev_utils.plotImagesBBoxes import plotFITSImageWithBoundingBoxes

from model.CustomMaskRCNN import CustomMaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights

import torchvision

device = get_device()




def eval_single_epoch(model, val_loader):
    # accs, losses = [], []
    # with torch.no_grad():
    #     model.eval()
    #     for x, y in val_loader:
    #         x, y = x.to(device), y.to(device)
    #         y_ = model(x)
    #         loss = F.cross_entropy(y_, y)
    #         acc = accuracy(y, y_)
    #         losses.append(loss.item())
    #         accs.append(acc.item())
    # return np.mean(losses), np.mean(accs)
    pass


def train_model(config, tempdir):

    data_transforms = A.Compose([A.AtLeastOneBBoxRandomCrop(width=512, height=512), A.RandomRotate90(p=1), A.ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'],filter_invalid_bboxes=True))

    joan_oro_dataset = TelescopeDataset(data_path=config["train_data_path"], cache_dir=tempdir, transform=data_transforms, device=device, loadasrgb=config["load_data_as_rgb"])

    train_dataset, val_dataset = torch.utils.data.random_split(joan_oro_dataset, [0.82, 0.18])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=custom_collate_fn) # TODO add validation

    backbone = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.DEFAULT)
    anchor_generator  = torchvision.models.detection.anchor_utils.AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3', '4'],
        output_size=7,
        sampling_ratio=2
    )
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3', '4'],
        output_size=14,
        sampling_ratio=2
    )

    model = CustomMaskRCNN(backbone, 
        backbone_trainable=True,
        num_classes=joan_oro_dataset.num_classes,
        rpn_anchor_generator=anchor_generator, 
        box_roi_pooler=roi_pooler,
        mask_roi_pooler=mask_roi_pooler)
    
    if config["load_data_as_rgb"]:
        model.SetToRGB()  # Set the model to accept RGB images
    else:
        model.SetToGrayscale()  # Set the model to accept grayscale images
    

    model.setTrainableLayers("roi_heads2_first_layer")  # Set the first layer of the ROI heads to be trainable

    model = model.train()
    optimizer = torch.optim.Adam(list(model.backbone.parameters()) + list(model.roi_heads.box_predictor.parameters()), lr=1e-4, weight_decay=0.001)

    loss_history = defaultdict(list)
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        for _, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"):
            model.train()
            predictions, loss_dict = train_single_epoch(model, images, targets, optimizer, device)

            for k, v in loss_dict.items():
                loss_history[k].append(v.item())

    # save_model(model)
    plot_losses(loss_history, fname="train_loss.png", save_plot=True)

def train_single_epoch(model, images, targets, optimizer, device):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer.zero_grad()

    predictions, loss_dict = model(images, targets)
    loss = sum(loss for loss in loss_dict.values())

    loss.backward()
    optimizer.step()
    
    return predictions, loss_dict

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

    # model = load_model(device, load_weights =True)
    # model.eval()

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
    # mode = read_arguments()
    # print(f'mode', mode)

    config = {
        "lr": 1e-3,
        "batch_size": 2,
        "epochs": 15,
        "train_data_path": "..\\images1000",
        "load_data_as_rgb": True,
        # "train_data_path": "C:\\Users\\RaulOnrubiaIbanez\\OneDrive - Zenithal Blue Technologies S.L.U\\Personal\\UPC\\JOData",
        # "train_data_path": "data_full",
    }
    mode = Mode.TRAIN

    with tempfile.TemporaryDirectory() as tempdir:
        if mode == Mode.TRAIN:
            train_model(config, tempdir)

        elif mode == Mode.INFER:
            inference('data_inference', tempdir)

if __name__ == "__main__":
    main()
