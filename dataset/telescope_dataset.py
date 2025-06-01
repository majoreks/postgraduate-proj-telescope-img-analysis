import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from dataset.file_path import DataType, FilePath, get_basename_prefix
from dataset.get_strided_vector import getStridedVector
from dataset.image_reader import read_image
from dataset.labels_reader import Y_MAX_KEY, X_MAX_KEY, X_MIN_KEY, Y_MIN_KEY,CLASS_KEY,COORDINATES_KEYS,read_labels

import albumentations as A

IMAGE_LENGTH = 4108
IMAGE_WIDTH = 4096

class TelescopeDataset(Dataset):
    def __init__(self, data_path, cache_dir, device: torch.device, transform : A.core.composition.Compose = None, Npixels=512+1):
        super().__init__()

        self.device = device

        self.data_path = data_path
        self.cache_dir = cache_dir
        self.transform = transform
        self.crop_size = Npixels

        image_paths = list(Path(self.data_path).rglob('*_V_imc.fits.gz'))
        label_paths = list(Path(self.data_path).rglob('*_V_imc_trl.dat'))

        image_map = {get_basename_prefix(p): p for p in image_paths}
        label_map = {get_basename_prefix(p): p for p in label_paths}
        common_keys = sorted(set(image_map.keys()) & set(label_map.keys()))

        self.images_list = [
            str(FilePath(key, DataType.IMAGE)) for key in common_keys
        ]
        self.labels_list = [
            str(FilePath(key, DataType.LABEL)) for key in common_keys
        ]

        lentgh_positions = getStridedVector(IMAGE_LENGTH, Npixels)
        width_positions = getStridedVector(IMAGE_WIDTH, Npixels)

        self.Nlength = len(lentgh_positions)
        self.Nwidth = len(width_positions)
        self.Nsubimages = self.Nlength * self.Nwidth

        lenmesh, lonmesh = np.meshgrid(range(self.Nlength), range(self.Nwidth))
        posindexes = np.array([lenmesh.flatten(), lonmesh.flatten()]).T

        self.coordinates = []
        self.transform = transform

        for ii in range(posindexes.shape[0]):
            x = lentgh_positions[posindexes[ii][0]]
            y = width_positions[posindexes[ii][1]]
            elem = [x[0],  x[1], y[0], y[1]]
            self.coordinates.append(elem)

    def __len__(self):
        return len(self.images_list)*self.Nsubimages
    
    def __getitem__(self, idx):
        image_idx = idx // self.Nsubimages
        subimage_idx = idx % self.Nsubimages

        image_path = Path(self.data_path, self.images_list[image_idx])
        label_path = Path(self.data_path, self.labels_list[image_idx])

        coords = self.coordinates[subimage_idx]

        image_data = read_image(image_path, self.cache_dir)[coords[0]:coords[1], coords[2]:coords[3]]
        labels_data = read_labels(label_path)

        labels_data = labels_data[
            (labels_data[X_MIN_KEY] >= coords[2]) & (labels_data[X_MIN_KEY] <= coords[3]) &
            (labels_data[Y_MIN_KEY] >= coords[0]) & (labels_data[Y_MIN_KEY] <= coords[1])
        ]

        labels_data[X_MIN_KEY] -= coords[2]
        labels_data[Y_MIN_KEY] -= coords[0]
        labels_data[X_MAX_KEY] -= coords[2]      # x_max
        labels_data[Y_MAX_KEY] -= coords[0]      # y_max

        within_x = (labels_data[X_MAX_KEY] > 0) & (labels_data[X_MIN_KEY] < self.crop_size)
        within_y = (labels_data[Y_MAX_KEY] > 0) & (labels_data[Y_MIN_KEY] < self.crop_size)
        labels_data = labels_data[within_x & within_y]

        label_data = np.array(labels_data[CLASS_KEY])
        bbox_data = np.array(labels_data[COORDINATES_KEYS], dtype=np.float32)

        if self.transform:
            transformed = self.transform(image=image_data, bboxes=bbox_data.tolist(), labels=label_data.tolist())
            image_data = transformed['image']
            bbox_data = transformed['bboxes']
            label_data = transformed['labels']

        # bbox_data /= self.crop_size
    
        targets = {"boxes": torch.tensor(bbox_data, dtype=torch.float32),
                   "labels": torch.tensor(label_data, dtype=torch.int64)}

        return image_data, targets
        