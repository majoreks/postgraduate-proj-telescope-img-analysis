import tempfile
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw
from tqdm import tqdm
from dataset.dataloader import custom_collate_fn
from dataset.telescope_dataset import TelescopeDataset

import albumentations as A

from model.FastRCNNPredictor import FastRCNNPredictor
from model.FasterRCNN import FasterRCNN
from model.TwoMLPHead import TwoMLPHead
from config.device import get_device
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

device = get_device()

# from model import MyModel
# from utils import accuracy
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np

def train_single_epoch(model, train_loader, optimizer):
    # model.train()
    # accs, losses = [], []
    # for x, y in train_loader:
    #     optimizer.zero_grad()
    #     x, y = x.to(device), y.to(device)
    #     y_ = model(x)
    #     loss = F.cross_entropy(y_, y)
    #     loss.backward()
    #     optimizer.step()
    #     acc = accuracy(y, y_)
    #     losses.append(loss.item())
    #     accs.append(acc.item())
    # return np.mean(losses), np.mean(accs)
    pass


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


def train_model(config):

    # data_transforms = transforms.Compose([transforms.ToTensor()])

    data_transforms = A.Compose([A.RandomRotate90(p=1), A.ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True))

    with tempfile.TemporaryDirectory() as tempdir:
        print(tempdir)
        joan_oro_dataset = TelescopeDataset(data_path = config["data_path"], cache_dir=tempdir, transform=data_transforms, device=device)
        # im, lab = joan_oro_dataset.__getitem__(2)  # Test if the dataset is working
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(joan_oro_dataset, [0.7, 0.15, 0.15])
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], collate_fn=custom_collate_fn)

        # x, y = next(iter(train_loader))
        # print(y[0])
        # print(x[0])
        # return
    
        backbone = resnet_fpn_backbone("resnet50", pretrained=True)
        model = FasterRCNN(backbone)

        in_size = 256*7*7
        representation_size = 1024
        num_classes = 91
        model.roi_heads.box_head = TwoMLPHead(in_size, representation_size)
        model.roi_heads.box_predictor = FastRCNNPredictor(representation_size, num_classes)

        url = "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
        state_dict =  torch.hub.load_state_dict_from_url(url)
        model.load_state_dict(state_dict)

        # old_conv = model.backbone.body.conv1
        # new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        # model.backbone.body.conv1 = new_conv

        model.to(device).eval()

        
        num_classes = 3 # stars, galaxies + background

        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Set model to training mode and move to device
        model = model.train().to(device)

        # Create optimizer for ONLY the new box_predictor module
        optimizer = torch.optim.Adam(
            model.roi_heads.box_predictor.parameters(),
            lr=1e-4,  # default is usually 1e-3, you can tune this
            weight_decay=0.0001
        )

        # x, y = next(iter(train_loader))
        # print(y)
        epoch_losses = []

        model.train()
        for epoch in range(config['epochs']):
            print(f"\nEpoch {epoch+1}/{config['epochs']}")

            running_loss = 0.0

            for i, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                predictions, loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # if i%100 == 0:
                #     loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}
                #     print(f"[{i}/{len(train_loader)}] loss: {loss_dict_printable}")

            avg_epoch_loss = running_loss / len(train_loader)
            epoch_losses.append(avg_epoch_loss)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('output/train_loss.png', dpi=400)

        # optimizer = optim.Adam(my_model.parameters(), config["lr"])
        # for epoch in range(config["epochs"]):
        #     loss, acc = train_single_epoch(my_model, train_loader, optimizer)
        #     print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        #     loss, acc = eval_single_epoch(my_model, val_loader)
        #     print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        
        # loss, acc = eval_single_epoch(my_model, test_loader)
        # print(f"Test loss={loss:.2f} acc={acc:.2f}")

        # return my_model

        # Put model in eval mode

        model = model.eval()

        # Pick an image from the dataset
        image, target = joan_oro_dataset[0]  # or any index you'd like

        # Run inference
        with torch.no_grad():
            detections = model([image.to(device)])[0]

        # NMS and thresholding
        iou_threshold = 0.2
        score_threshold = 0.4

        keep_idx = torchvision.ops.nms(detections["boxes"], detections["scores"], iou_threshold)

        boxes = [b for i, b in enumerate(detections["boxes"]) if i in keep_idx and detections["scores"][i] > score_threshold]
        scores = [s for i, s in enumerate(detections["scores"]) if i in keep_idx and s > score_threshold]
        labels = [l for i, l in enumerate(detections["labels"]) if i in keep_idx and detections["scores"][i] > score_threshold]

        # Class label mapping
        idx2label = {
            1: "star",
            2: "galaxy"
        }

        # Visualize results
        im = to_pil_image(image.cpu())
        draw = ImageDraw.Draw(im)

        for box, score, label in zip(boxes, scores, labels):
            coords = box.cpu().tolist()
            draw.rectangle(coords, outline="red", width=2)
            text = f"{idx2label.get(label.item(), 'unknown')} {score.item()*100:.1f}%"
            draw.text((coords[0], coords[1] - 10), text, fill="white")

        # Show image
        plt.imshow(im)
        plt.axis('off')
        plt.savefig('output/test.png', dpi=400)
        pass


if __name__ == "__main__":

    config = {
        "lr": 1e-3,
        "batch_size": 8,
        "epochs": 5,
        "data_path": "data_full"
    }

    my_model = train_model(config)

    