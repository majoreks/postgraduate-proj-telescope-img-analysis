import os
import numpy as np
from torch.utils.data import Dataset
from astropy.io import fits
from pathlib import Path
from dataset.file_path import DataType, FilePath, get_basename_prefix
from dataset.get_strided_vector import getStridedVector
from dataset.labels_reader import HEIGHT_KEY, WIDTH_KEY, X_KEY, Y_KEY, read_labels

class TelescopeDataset(Dataset):
    def __init__(self, data_path, transform=None, Npixels=512):
        super().__init__()

        self.data_path = data_path
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

        labels_data = read_labels(label_path)

        labels_data = labels_data[
            (labels_data[X_KEY] >= coords[2]) & (labels_data[X_KEY] <= coords[3]) &
            (labels_data[Y_KEY] >= coords[0]) & (labels_data[Y_KEY] <= coords[1])
        ].copy()

        labels_data[X_KEY] -= coords[2]
        labels_data[Y_KEY] -= coords[0]

        labels_data[X_KEY] /= self.crop_size
        labels_data[Y_KEY] /= self.crop_size
        labels_data[WIDTH_KEY] /= self.crop_size
        labels_data[HEIGHT_KEY] /= self.crop_size
        
        if self.transform:
            image_data = self.transform(image_data)
            ## This is going to be tricky!!!!
            # filtered_labels = self.transform(filtered_labels)

        labels_data = labels_data.to_dict(orient='records')
        return image_data, labels_data
        