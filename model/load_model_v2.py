from typing import Callable
import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models import resnet50, resnet34, resnet101, resnet18, ResNet34_Weights, ResNet101_Weights, ResNet50_Weights, WeightsEnum, ResNet18_Weights, ResNet34_Weights
from model.FastRCNNConvFCHead import FastRCNNConvFCHead
from model.FastRCNNPredictor import FastRCNNPredictor
from model.FasterRCNN import FasterRCNN
from model.TwoMLPHead import TwoMLPHead
from model.model_reader import read_model

save_path = 'output/model/model_weights.pt'


in_size = 256*7*7
representation_size = 1024
num_classes_pretrained = 91
num_classes = 2 # object of interest + background

def resnet_loader(resnet_type: str | None) -> tuple[Callable, WeightsEnum]:
    if resnet_type is None or resnet_type == 'resnet50':
        return resnet50, ResNet50_Weights
    if resnet_type == 'resnet18':
        return resnet18, ResNet18_Weights
    if resnet_type == 'resnet34':
        return resnet34, ResNet34_Weights
    if resnet_type == 'resnet101':
        return resnet101, ResNet101_Weights

def load_model_v2(device: torch.device, config:dict, load_weights: bool = False, weights_path: str = None, resnet_type: str | None = None) -> nn.Module:
    print(f'loading v2 model')
    
    box_detections_per_img= config['box_detections_per_img']
    nms_threshold = config['nms_threshold']

    resnet, weights = resnet_loader(resnet_type)

    print(f'backbone {resnet_type} | using pretrained weights {weights}')
    backbone = resnet(weights=weights, progress=True)

    # trainable_backbone_layers = _validate_trainable_layers(True, None, 5, 3)
    trainable_backbone_layers = 5
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, norm_layer=nn.BatchNorm2d)
    model = FasterRCNN(backbone, box_detections_per_img=box_detections_per_img, conv_depth=2)

    model.roi_heads.box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(representation_size, num_classes if load_weights else num_classes_pretrained)

    if load_weights:
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.backbone.body.conv1 = new_conv

        model = read_model(model, device, weights_path)
        model = model.to(device)

        return model

    if resnet_type is None:
        print(f'loading pretrained weights for the model')
        url = "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
        state_dict =  torch.hub.load_state_dict_from_url(url)
        model.load_state_dict(state_dict)

    # substitute first layer with one that accepts 1 channel image
    old_conv = model.backbone.body.conv1
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    model.backbone.body.conv1 = new_conv

    for param in model.parameters():
        param.requires_grad = True

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.nms_thresh = nms_threshold

    model = model.to(device)
    
    return model

def read_model_v2(model: nn.Module, device: torch.device, path: str | None = None) -> nn.Module:
    params = torch.load(save_path if path is None else path, map_location=device)
    model.load_state_dict(params)
    return model