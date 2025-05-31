import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataloader import custom_collate_fn
from dataset.telescope_dataset import TelescopeDataset

import albumentations as A

from model.FastRCNNPredictor import FastRCNNPredictor
from model.FasterRCNN import FasterRCNN
from model.TwoMLPHead import TwoMLPHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


# from model import MyModel
# from utils import accuracy
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    data_transforms = A.Compose([A.RandomRotate90(p=1), A.ToTensorV2()], bbox_params=A.BboxParams(format='coco', label_fields=['labels'], filter_invalid_bboxes=True))

    with tempfile.TemporaryDirectory() as tempdir:
        print(tempdir)
        joan_oro_dataset = TelescopeDataset(data_path = config["data_path"], cache_dir=tempdir, transform=data_transforms)
        # im, lab = joan_oro_dataset.__getitem__(2)  # Test if the dataset is working
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(joan_oro_dataset, [0.7, 0.15, 0.15])
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], collate_fn=custom_collate_fn)


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

        old_conv = model.backbone.body.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        model.backbone.body.conv1 = new_conv

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

        model.train()
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions, loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%2 == 0:
                loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}
                print(f"[{i}/{len(train_loader)}] loss: {loss_dict_printable}")

    # optimizer = optim.Adam(my_model.parameters(), config["lr"])
    # for epoch in range(config["epochs"]):
    #     loss, acc = train_single_epoch(my_model, train_loader, optimizer)
    #     print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
    #     loss, acc = eval_single_epoch(my_model, val_loader)
    #     print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
    
    # loss, acc = eval_single_epoch(my_model, test_loader)
    # print(f"Test loss={loss:.2f} acc={acc:.2f}")

    # return my_model
    pass


if __name__ == "__main__":

    config = {
        "lr": 1e-3,
        "batch_size": 8,
        "epochs": 5,
        "data_path": "data"
    }

    my_model = train_model(config)

    