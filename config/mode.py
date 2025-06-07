from enum import Enum

class Mode(Enum):
    TRAIN = "train"
    INFER = "infer"

    def __str__(self):
        return self.value  # For argparse to print readable choices