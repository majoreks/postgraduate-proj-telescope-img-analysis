import os
from getStridedVector import getStridedVector
from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np


class MyDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None, Npixels=512):
        super().__init__()
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform

        self.images_list = os.listdir(self.images_path)
        self.labels_list = os.listdir(self.labels_path)

        images_list = [image.split("_V_")[0] for image in self.images_list]
        labels_list = [label.split("_V_")[0] for label in self.labels_list]

        combined_list = [image for image in images_list if image in labels_list]

        image_str = "_V_imc.fits.gz"
        label_str = "_V_imc_trl.dat"

        self.images_list = [image + image_str for image in combined_list]
        self.labels_list = [label + label_str for label in combined_list]

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

        image_path = os.path.join(self.images_path, self.images_list[image_idx])
        label_path = os.path.join(self.labels_path, self.labels_list[image_idx])

        coords = self.coordinates[subimage_idx]

        # Load the image and label data
        with fits.open(image_path) as hdul:
            image_data = hdul[0].data[coords[0]:coords[1], coords[2]:coords[3]]
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
        