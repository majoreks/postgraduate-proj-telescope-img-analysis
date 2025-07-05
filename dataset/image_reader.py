from pathlib import Path
import numpy as np
from astropy.io import fits

SUFFIX_NPY = ".npy"
SUFFIX_SCLAE = ".scale"

def read_image(image_path: str, cache_dir: str) -> tuple[np.ndarray, float]: 
    image_cached_path = Path(f"{cache_dir}/{Path(image_path).name}").with_suffix(SUFFIX_NPY)
    pixel_scale_cached_path = Path(f"{cache_dir}/{Path(image_path).name}").with_suffix(SUFFIX_SCLAE + SUFFIX_NPY)

    if image_cached_path.exists() and pixel_scale_cached_path.exists():
        return np.load(image_cached_path), float(np.load(pixel_scale_cached_path))
    
    with fits.open(image_path) as hdul:
        image_data = extract_data(hdul)
        pixel_scale = extract_pixel_scale(hdul)

        save_cache(image_cached_path, image_data)
        save_cache(pixel_scale_cached_path, pixel_scale)

        return image_data, pixel_scale

def save_cache(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)  
    np.save(path, data)

def extract_data(hdul):
    return hdul[0].data.astype(np.float32)

def extract_pixel_scale(hdul):
    hdr = hdul[0].header
    cdelt1 = abs(hdr.get('CD1_1', None))
    cdelt2 = abs(hdr.get('CD2_2', None))
    if cdelt1 and cdelt2:
        return (abs(cdelt1) + abs(cdelt2)) / 2
    else:
        raise ValueError("No se encontró CDELT1/CDELT2 en el header del FITS")