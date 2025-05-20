import torch
from torch.utils.data import DataLoader
from dataset.datasetClass import telescopeDataset
from torchvision import transforms

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

    data_transforms = transforms.Compose([transforms.ToTensor()])
    joan_oro_dataset = telescopeDataset(data_path = config["data_path"], transform=data_transforms)
    joan_oro_dataset.__getitem__(0)
    random_seed = 42
    torch.manual_seed(random_seed)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(joan_oro_dataset, [0.7, 0.15, 0.15])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    # my_model = MyModel().to(device)

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
        "data_path": "./data"
    }

    my_model = train_model(config)
    