import torchvision as torchvision
from torchvision.models.detection import MaskRCNN
import torch

class CustomMaskRCNN(MaskRCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_first_layer = self.backbone.body.conv1
        self.loadImagesAsRGB = True  # Flag to indicate if we need to replicate image channels
        pass

    def setTrainableLayers(self, mode):
        if mode == "all":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif mode == "backbone":
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.roi_heads.parameters():
                param.requires_grad = False
        elif mode == "roi_heads":
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.roi_heads.parameters():
                param.requires_grad = True
        elif mode == "roi_heads2_first_layer":
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.body.conv1.requires_grad = True  # Keep the first layer trainable

            for param in self.roi_heads.parameters():
                param.requires_grad = True
            

        elif mode == "none":
            for param in self.parameters():
                param.requires_grad = False
        else:
            raise ValueError("Invalid mode. Use 'all', 'backbone', or 'roi_heads'.")


    def SetToRGB(self):
        # This resets the first layer to its original state
        self.loadImagesAsRGB = True
        self.backbone.body.conv1 = self.original_first_layer


    def SetToGrayscale(self):
        
        # Change first conv layer to accept 1 channel (grayscale)
        self.loadImagesAsRGB = False

        in_channels = 1
        out_channels = self.backbone.body.conv1.out_channels
        kernel_size = self.backbone.body.conv1.kernel_size
        stride = self.backbone.body.conv1.stride
        padding = self.backbone.body.conv1.padding
        bias = self.backbone.body.conv1.bias is not None

        new_conv = torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias
        )
        with torch.no_grad():
            new_conv.weight[:, 0] = self.backbone.body.conv1.weight.mean(dim=1)
            if bias:
                new_conv.bias = self.backbone.body.conv1.bias
            self.backbone.body.conv1 = new_conv
        pass

        
        