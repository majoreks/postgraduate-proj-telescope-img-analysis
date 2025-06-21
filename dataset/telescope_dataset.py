import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from dataset.file_path import DataType, FilePath, get_basename_prefix
from dataset.image_reader import read_image
from dataset.labels_reader import (
    CLASS_KEY, COORDINATES_KEYS, read_labels
)
import albumentations as A

class TelescopeDataset(Dataset):
    def __init__(self, data_path, cache_dir, device: torch.device, transform: A.core.composition.Compose = None):
        super().__init__()

        self.device = device
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.transform = transform

        image_paths = list(Path(self.data_path).rglob('*_imc.fits.gz'))
        label_paths = [
            p for p in Path(self.data_path).rglob('*_imc_trl.dat')
            if p.stat().st_size > 1344
        ]
        empty_paths = [
            p for p in Path(self.data_path).rglob('*_imc_trl.dat')
            if p.stat().st_size == 1344
        ]

        print("🔍 Total imágenes encontradas:", len(image_paths))
        print("🔍 Total etiquetas encontradas:", len(label_paths))
        print("🔍 Total etiqueta vacías:", len(empty_paths))


        image_map = {get_basename_prefix(p): p for p in image_paths}
        label_map = {get_basename_prefix(p): p for p in label_paths}
        common_keys = sorted(set(image_map.keys()) & set(label_map.keys()))

        self.images_list = [str(FilePath(key, DataType.IMAGE)) for key in common_keys]
        self.labels_list = [str(FilePath(key, DataType.LABEL)) for key in common_keys]
        
        empty_image_map = {get_basename_prefix(p): p for p in image_paths}
        empty_label_map = {get_basename_prefix(p): p for p in empty_paths}
        empty_common_keys = sorted(set(empty_image_map.keys()) & set(empty_label_map.keys()))

        self.empty_images_list = [str(FilePath(key, DataType.IMAGE)) for key in empty_common_keys]
        self.empty_labels_list = [str(FilePath(key, DataType.LABEL)) for key in empty_common_keys]
        
        pass

    def __len__(self):
        return len(self.images_list)

    def move_empty_to_folder(self, folder_path: str):
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)

        for image_path, label_path in zip(self.empty_images_list, self.empty_labels_list):
            image_file = Path(self.data_path, image_path)
            label_file = Path(self.data_path, label_path)

            if image_file.exists() and label_file.exists():
                image_file.rename(folder_path / image_file.name)
                label_file.rename(folder_path / label_file.name)
            else:
                print(f"File not found: {image_file} or {label_file}")
        self.empty_images_list = []
        self.empty_labels_list = []

    def move_dataset_to_folder(self, folder_path: str, indexes=None):
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)

        if indexes is None:
            indexes = range(len(self.images_list))

        indexes = list(indexes)

        for image_path, label_path in zip([self.images_list[i] for i in indexes], [self.labels_list[i] for i in indexes]):
            image_file = Path(self.data_path, image_path)
            label_file = Path(self.data_path, label_path)

            if image_file.exists() and label_file.exists():
                image_file.rename(folder_path / image_file.name)
                label_file.rename(folder_path / label_file.name)
            else:
                print(f"File not found: {image_file} or {label_file}")


    def __getitem__(self, idx):
        image_path = Path(self.data_path, self.images_list[idx])
        label_path = Path(self.data_path, self.labels_list[idx])

        image_data = read_image(image_path, self.cache_dir)  # Shape: [H, W]
        labels_data = read_labels(label_path)

        label_data = np.array(labels_data[CLASS_KEY])
        bbox_data = np.array(labels_data[COORDINATES_KEYS], dtype=np.float32)

        image_data = np.expand_dims(image_data, axis=2)  # [H, W, 1]

        if self.transform:
            transformed = self.transform(
                image=image_data,
                bboxes=bbox_data.tolist(),
                labels=label_data.tolist()
            )
            image_data = transformed['image']
            bbox_data = transformed['bboxes']
            label_data = transformed['labels']

        targets = {
            "boxes": torch.tensor(bbox_data, dtype=torch.float32),
            "labels": torch.tensor(label_data, dtype=torch.int64)
        }

        return image_data, targets