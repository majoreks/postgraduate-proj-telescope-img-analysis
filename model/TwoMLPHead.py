import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import TwoMLPHead


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_size (int): number of inputs
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_size, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)  # Flatten to shape [batch, in_size]
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x