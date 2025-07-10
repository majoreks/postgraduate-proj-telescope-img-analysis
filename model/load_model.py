import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from model.FastRCNNPredictor import FastRCNNPredictor
from model.FasterRCNN import FasterRCNN
from model.TwoMLPHead import TwoMLPHead
from model.model_reader import read_model

in_size = 256*7*7
representation_size = 1024
num_classes_pretrained = 91
num_classes = 2 # object of interest + background

def load_model(device: torch.device,config: dict, nms_threshold: float, load_weights: bool = False, weights_path: str = None) -> nn.Module:

    box_detections_per_img= config['box_detections_per_img']

    backbone = resnet_fpn_backbone("resnet50", pretrained=not load_weights)
    model = FasterRCNN(backbone, box_detections_per_img=box_detections_per_img)

    model.roi_heads.box_head = TwoMLPHead(in_size, representation_size)
    model.roi_heads.box_predictor = FastRCNNPredictor(representation_size, num_classes if load_weights else num_classes_pretrained)

    if load_weights:
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.backbone.body.conv1 = new_conv

        model = read_model(model, device, weights_path)
        model = model.to(device)

        return model

    url = "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    state_dict =  torch.hub.load_state_dict_from_url(url)
    model.load_state_dict(state_dict)

    # substitute first layer with one that accepts 1 channel image
    old_conv = model.backbone.body.conv1
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    model.backbone.body.conv1 = new_conv

    for name, param in model.backbone.named_parameters():
        if name.startswith("body.conv1") or name.startswith("body.layer1"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.nms_thresh = nms_threshold

    model = model.to(device)
    
    return model