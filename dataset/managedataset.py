import os
import torch
from dataset.telescope_dataset import TelescopeDataset

def split_dataset(config: dict, temp_dir, device) -> None:

    data_path = config["data_path"]

    gz_files = []
    for root, _, files in os.walk(data_path):
        gz_files.extend([os.path.join(root, f) for f in files if f.endswith('.gz')])
    dat_files = []
    for root, _, files in os.walk(data_path):
        dat_files.extend([os.path.join(root, f) for f in files if f.endswith('.dat')])

    for file_path in gz_files:
        dest_path = os.path.join(data_path, os.path.basename(file_path))
        if file_path != dest_path:
            os.rename(file_path, dest_path)

    for file_path in dat_files:
        dest_path = os.path.join(data_path, os.path.basename(file_path))
        if file_path != dest_path:
            os.rename(file_path, dest_path)

    # Remove all empty folders recursively under data_path
    for root, dirs, files in os.walk(data_path, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

    joan_oro_dataset = TelescopeDataset(data_path=data_path, transform=None, device=device, cache_dir=temp_dir)
    joan_oro_dataset.move_empty_to_folder(os.path.join(config['data_path'], "metadataless_dataset"))
        
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(joan_oro_dataset, config['train_test_split'])
    train_dataset.dataset.move_dataset_to_folder(os.path.join(config['data_path'], "train_dataset"), indexes=train_dataset.indices)
    test_dataset.dataset.move_dataset_to_folder(os.path.join(config['data_path'], "test_dataset"), indexes=test_dataset.indices)

def check_and_split(config,temp_dir, device):

    if config["allow_splitting"] == True:
        data_path = config["data_path"]
        print("Listing folders in:", data_path)
        need_to_split = True
        if os.path.exists(data_path):
            if len(os.listdir(data_path)) == 3:
                folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(config["data_path"], f))]
                if set(folders) == {"metadataless_dataset", "test_dataset", "train_dataset"}:
                    need_to_split = False
                    print("Dataset already split into train and test folders.")
        
        if need_to_split == True:
            print("Dataset not split, merging all files and splitting now...")
            split_dataset(config,temp_dir=temp_dir, device=device)