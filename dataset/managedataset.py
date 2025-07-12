import os
import torch
from dataset.telescope_dataset import TelescopeDataset
from pathlib import Path

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

def crop_dataset(config: dict, source_folder) -> None:
    from astropy.io import fits

    crop_size = config["crop_size"]
    source_folder = Path(os.path.join(config['data_path'], source_folder))
    cropped_folder = Path(f"{source_folder}_cropped")

    if not os.path.exists(cropped_folder):
        os.makedirs(cropped_folder)

    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.fits.gz'):
                file_path = os.path.join(root, file)
                with fits.open(file_path) as hdul:

                    # Crop the image to the desired size
                    height, width = hdul[0].data.shape
                    height_vector = getStridedVector(height, crop_size)
                    width_vector = getStridedVector(width, crop_size)

                    Nsubimages = len(height_vector) * len(width_vector)
                    heightmesh, widthmesh = np.meshgrid(range(len(height_vector)), range(len(width_vector)))
                    posindexes = np.array([heightmesh.flatten(), widthmesh.flatten()]).T

                    datfilename = Path(os.path.join(source_folder, file[:-8] + "_trl.dat"))
                    if datfilename.exists():
                        datfilnamepresent = True
                        # Read the first 14 lines of the .dat file as text

                        hdr = hdul[0].header
                        cdelt1 = abs(hdr.get('CD1_1', None))
                        cdelt2 = abs(hdr.get('CD2_2', None))
                        pixel_scale = (abs(cdelt1) + abs(cdelt2)) / 2 if cdelt1 and cdelt2 else None

                        with open(datfilename, 'r') as datfile:
                            first_14_lines = [datfile.readline() for _ in range(14)]
                            datfile_lines = datfile.readlines()
                        labels = read_labels(datfilename, pixel_scale=pixel_scale)

                    else:
                        datfilnamepresent = False

                    for n in range(Nsubimages):
                        heights = height_vector[posindexes[n, 0]]
                        widths = width_vector[posindexes[n, 1]]

                        cropped_image = hdul[0].data[heights[0]:heights[1], widths[0]:widths[1]]
                        header = hdul[0].header.copy()
                        header.set("CROPSIZE", crop_size, "Size of the cropped image")
                        header.set("NSUBIMG", Nsubimages, "Total number of subimages")
                        header.set("CROPPOS", n, "Number of the cropped image")
                        header.set("HEIGHT0", heights[0], "Starting height of the cropped image")
                        header.set("HEIGHT1", heights[1], "Ending height of the cropped image")
                        header.set("WIDTH0", widths[0], "Starting width of the cropped image")
                        header.set("WIDTH1", widths[1], "Ending width of the cropped image")
                        header.set("ORIGINAL", file, "Original file name")

                        new_labels = []
                        for line in datfile_lines:
                            parts = line.split()
                            xy = (float(parts[0]), float(parts[1]))
                            if (widths[0] <= xy[0] <= widths[1]) and (heights[0] <= xy[1] <= heights[1]):
                                parts[0] = str(xy[0] - widths[0])
                                parts[1] = str(xy[1] - heights[0])
                                new_labels.append('  '.join(parts)+ '\n')


                        if len(new_labels) > 0:

                            fits.writeto(Path(os.path.join(cropped_folder, f"{file[:-14]}_{n}_U_imc.fits.gz")), cropped_image,header, overwrite=True)
                            if datfilnamepresent:
                                # Get the index elements in datfile_lines
                                # metadata = first_14_lines + [line for i, line in enumerate(datfile_lines) if i in indexes]
                                metadata = first_14_lines + [line for i, line in enumerate(new_labels)]
                                cropped_datfile_path = Path(os.path.join(cropped_folder, f"{file[:-14]}_{n}_U_imc_trl.dat"))
                                with open(cropped_datfile_path, 'w', newline='\n') as cropped_datfile:
                                    cropped_datfile.writelines(metadata)


def check_and_split(config,temp_dir, device):

    if config["allow_splitting"] == True:
        data_path = config["data_path"]
        print("Listing folders in:", data_path)
        need_to_split = True
        need_to_crop = False
        if os.path.exists(data_path):
            if len(os.listdir(data_path)) == 3:
                folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(config["data_path"], f))]
                if set(folders) == {"metadataless_dataset", "test_dataset", "train_dataset"}:
                    need_to_split = False
                    print("Dataset already split into train and test folders.")
        
        if need_to_split == True:
            print("Dataset not split, merging all files and splitting now...")
            split_dataset(config,temp_dir=temp_dir, device=device)
            need_to_crop = True

        if need_to_crop == True:

            crop_dataset(config, "train_dataset")
            crop_dataset(config, "test_dataset")
            crop_dataset(config, "metadataless_dataset")