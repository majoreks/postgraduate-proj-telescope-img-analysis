import numpy as np
from astropy.io import fits


def read_image(image_path: str) -> np.ndarray: 
    with fits.open(image_path) as hdul:
        return hdul[0].data.astype(np.float32)
