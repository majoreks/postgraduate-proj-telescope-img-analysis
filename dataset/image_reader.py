from pathlib import Path
import numpy as np
from astropy.io import fits



def read_image(image_path: str, cache_dir: str, loadasrgb=False) -> np.ndarray: 
    image_cached_path = Path(f"{cache_dir}/{Path(image_path).name}").with_suffix('.npy')

    if image_cached_path.exists():
        return np.load(image_cached_path)   
    
    with fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        if loadasrgb:
            data = np.repeat(np.expand_dims(data,axis=2), 3, axis=2)
        image_cached_path.parent.mkdir(parents=True, exist_ok=True)  
        np.save(image_cached_path, data)

        return data