from enum import Enum
import os
from getStridedVector import getStridedVector
from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np
from pathlib import Path

class DataType(Enum):
    LABEL = 1
    IMAGE = 2

class FilePath():
    __IMAGE_PATH_POSTFIX = "_V_imc.fits.gz"
    __LABEL_PATH_POSTFIX = "_V_imc_trl.dat"

    __IMAGE_PATH_DIR = "RED"
    __LABEL_PATH_DIR = "CAT"

    def __init__(self, key: str, type: DataType):
        self.__key = key
        self.__dataType = type

    def __str__(self) -> str:
        return self.path

    @property
    def path(self) -> str:
        return f"{self.__key[3:].split('.')[0]}/{self.__dir}/{self.__key}{self.__postfix}"

    @property
    def __postfix(self) -> str:
        if self.__dataType == DataType.IMAGE:
            return self.__IMAGE_PATH_POSTFIX
        elif self.__dataType == DataType.LABEL:
            return self.__LABEL_PATH_POSTFIX
        else:
            raise Exception("Unknown data type", self.__dataType)
        
    @property
    def __dir(self) -> str:
        if self.__dataType == DataType.IMAGE:
            return self.__IMAGE_PATH_DIR
        elif self.__dataType == DataType.LABEL:
            return self.__LABEL_PATH_DIR
        else:
            raise Exception("Unknown data type", self.__dataType)

def get_basename_prefix(path: Path) -> str:
    return path.name.split("_V_")[0]

def build_path(key: str, type: DataType) -> FilePath:
    return FilePath(key, type)

class TelescopeDataset(Dataset):
    def __init__(self, data_path, transform=None, Npixels=512):
        super().__init__()

        self.data_path = data_path
        self.transform = transform

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

        length = 4108
        width = 4096

        lentgh_positions = getStridedVector(length, Npixels)
        width_positions = getStridedVector(width, Npixels)

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

        image_path = os.path.join(self.data_path, self.images_list[image_idx])
        label_path = os.path.join(self.data_path, self.labels_list[image_idx])

        coords = self.coordinates[subimage_idx]

        # Load the image and label data
        with fits.open(image_path) as hdul:
            image_data = hdul[0].data.astype(np.float32)[coords[0]:coords[1], coords[2]:coords[3]]
            # image_data_cut = image_data

        with open(label_path, 'r') as f:
            labels_data = f.readlines()

        labels_data = np.array([[float(x) for x in numeric_string.split()] for numeric_string in labels_data[12:]])

        filtered_labels = labels_data[(labels_data[:,0]>=coords[2]) & (labels_data[:,0]<=coords[3]) & (labels_data[:,1]>=coords[0]) & (labels_data[:,1]<=coords[1])]
        filtered_labels[:,0] = filtered_labels[:,0] - coords[2]
        filtered_labels[:,1] = filtered_labels[:,1] - coords[0]
        
        if self.transform:
            image_data = self.transform(image_data)
            ## This is going to be tricky!!!!
            # filtered_labels = self.transform(filtered_labels)

        return image_data, filtered_labels
        