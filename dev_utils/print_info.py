from torch.utils.data import DataLoader

from dev_utils.plotImagesBBoxes import plotFITSImageWithBoundingBoxes

def print_info(loader: DataLoader) -> None:
    x, y = next(iter(loader))
    print(x[0].shape)
    print(len(y))
    print(y[0]["boxes"].shape)
    print(y[0]["labels"].shape)

    plotFITSImageWithBoundingBoxes(x[0], y[0], save_fig=True)
    
    return