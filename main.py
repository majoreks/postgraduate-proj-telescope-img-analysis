from collections import defaultdict
import tempfile
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.dataloader import custom_collate_fn
from dataset.telescope_dataset import TelescopeDataset

import albumentations as A

from model.FastRCNNPredictor import FastRCNNPredictor
from model.FasterRCNN import FasterRCNN
from model.TwoMLPHead import TwoMLPHead
from config.device import get_device
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from postprocess.plot_losses import plot_losses

device = get_device()

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

    data_transforms = A.Compose([A.AtLeastOneBBoxRandomCrop(width=512, height=512), A.RandomRotate90(p=1), A.ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True))

    with tempfile.TemporaryDirectory() as tempdir:
        joan_oro_dataset = TelescopeDataset(data_path = config["data_path"], cache_dir=tempdir, transform=data_transforms, device=device)

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

        # substitute first layer with one that accepts 1 channel image
        old_conv = model.backbone.body.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        model.backbone.body.conv1 = new_conv

        # setup first layer for retraining
        for name, param in model.backbone.named_parameters():
            if name.startswith("body.conv1") or name.startswith("body.layer1"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.to(device).eval()
        
        num_classes = 3 # stars, galaxies + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model = model.train().to(device)

        optimizer = torch.optim.Adam(list(model.backbone.parameters()) + list(model.roi_heads.box_predictor.parameters()), lr=1e-4, weight_decay=0.001)

        loss_history = defaultdict(list)
        for epoch in range(config['epochs']):
            print(f"\nEpoch {epoch+1}/{config['epochs']}")

            for i, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                predictions, loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for k, v in loss_dict.items():
                    loss_history[k].append(v.item())

        plot_losses(loss_history, fname="train_loss.png", save_plot=True)


if __name__ == "__main__":

    config = {
        "lr": 1e-3,
        "batch_size": 8,
        "epochs": 15,
        "data_path": "data_full"
    }

    my_model = train_model(config)
