import os
from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np
import pathlib

from .labels_reader import read_labels
import matplotlib.pyplot as plt

class telescopeDataset(Dataset):
    def __init__(self, data_path, transform=None, Npixels=512):
        super().__init__()

        self.data_path = data_path
        self.transform = transform



        self.images_list = (list(pathlib.Path(self.data_path).rglob('*_V_imc.fits.gz')))
        self.labels_list = (list(pathlib.Path(self.data_path).rglob('*_V_imc_trl.dat')))

        images_list = [str(image).split("_V_")[0].split("\\")[-1] for image in self.images_list]
        labels_list = [str(label).split("_V_")[0].split("\\")[-1] for label in self.labels_list]

        combined_list = [image for image in images_list if image in labels_list]

        image_str = "_V_imc.fits.gz"
        label_str = "_V_imc_trl.dat"

        self.images_list = [image[3:].split(".")[0] + "\\RED\\" + image + image_str for image in combined_list]
        self.labels_list = [label[3:].split(".")[0] + "\\CAT\\" + label + label_str for label in combined_list]

        self.rows = 4108
        self.cols = 4096

        self.Npixels = Npixels

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):

        image_path = os.path.join(self.data_path, self.images_list[idx])
        label_path = os.path.join(self.data_path, self.labels_list[idx])
        
        start_x = np.random.randint(0, self.cols-self.Npixels)
        start_y = np.random.randint(0, self.rows-self.Npixels)

        coords = [start_y, start_y + self.Npixels,start_x, start_x + self.Npixels]

        # Load the image and label data
        with fits.open(image_path) as hdul:
            image_data = hdul[0].data[coords[0]:coords[1], coords[2]:coords[3]]
        
        labels_data = read_labels(label_path)

        labels_data['x'] = labels_data['x'] - start_x
        labels_data['y'] = labels_data['y'] - start_y

        labels_data = labels_data[(labels_data['x']>=0) & (labels_data['x']<self.Npixels)]
        labels_data = labels_data[(labels_data['y']>=0) & (labels_data['y']<self.Npixels)]

        plt.figure(figsize=(10, 10))
        plt.imshow(image_data, cmap='gray')
        plt.scatter(labels_data['x'], labels_data['y'], c='red', s=10)
        plt.show()
        
        if self.transform:
            image_data = self.transform(image_data)
            ## This is going to be tricky!!!!
            # filtered_labels = self.transform(filtered_labels)

        return image_data, labels_data