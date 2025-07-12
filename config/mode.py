from enum import Enum

class Mode(Enum):
    TRAIN = "train"
    INFER = "infer"
    EXPERIMENT = "experiment"

    def __str__(self):
        return self.value  # For argparse to print readable choices