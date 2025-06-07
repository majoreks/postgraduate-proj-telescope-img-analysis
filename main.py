import torch
import tempfile
import albumentations as A
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from config.arg_reader import read_arguments
from config.mode import Mode
from dataset.dataloader import custom_collate_fn
from dataset.telescope_dataset import TelescopeDataset
from config.device import get_device
from model.load_model import load_model
from model.model_reader import save_model
from postprocess.plot_losses import plot_losses

device = get_device()

def train_single_epoch(model, images, targets, optimizer, device):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer.zero_grad()

    predictions, loss_dict = model(images, targets)
    loss = sum(loss for loss in loss_dict.values())

    loss.backward()
    optimizer.step()
    
    return predictions, loss_dict


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


def train_model(config, tempdir):

    data_transforms = A.Compose([A.AtLeastOneBBoxRandomCrop(width=512, height=512), A.RandomRotate90(p=1), A.ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True))

    joan_oro_dataset = TelescopeDataset(data_path=config["train_data_path"], cache_dir=tempdir, transform=data_transforms, device=device)

    train_dataset, val_dataset = torch.utils.data.random_split(joan_oro_dataset, [0.82, 0.18])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=custom_collate_fn) # TODO add validation

    model = load_model(device)

    model = model.train()
    optimizer = torch.optim.Adam(list(model.backbone.parameters()) + list(model.roi_heads.box_predictor.parameters()), lr=1e-4, weight_decay=0.001)

    loss_history = defaultdict(list)
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        for _, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"):
            model.train()
            predictions, loss_dict = train_single_epoch(model, images, targets, optimizer, device)

            for k, v in loss_dict.items():
                loss_history[k].append(v.item())

    save_model(model)
    plot_losses(loss_history, fname="train_loss.png", save_plot=True)

def inference(path, tempdir):
    data_transforms = A.Compose([A.ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], filter_invalid_bboxes=True))

    dataset = TelescopeDataset(data_path=path, cache_dir=tempdir, transform=data_transforms, device=device)
    test_loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)

    model = load_model(device, True)

    print(len(test_loader))
    print(model)

def main() -> None:
    mode = read_arguments()
    print(f'mode', mode)

    config = {
        "lr": 1e-4,
        "batch_size": 8,
        "epochs": 15,
        "train_data_path": "data_full"
    }

    with tempfile.TemporaryDirectory() as tempdir:
        if mode == Mode.TRAIN:
            train_model(config, tempdir)

        elif mode == Mode.INFER:
            inference('data_inference', tempdir)

if __name__ == "__main__":
    main()
