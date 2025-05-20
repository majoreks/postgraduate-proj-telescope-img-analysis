import pandas as pd

__all__ = ['read_labels']

dat_labels = [
    "X_IMAGE", "Y_IMAGE", "ALPHA_J2000", "DELTA_J2000",
    "MAG_AUTO", "MAGERR_AUTO", "FWHM_WORLD", "FLUX_RADIUS",
    "ELLIPTICITY", "THETA_WORLD", "THETA_J2000", "FLAGS"
]

def calculate_bbox(row, scale=1.0):
    x_center = row["X_IMAGE"]
    y_center = row["Y_IMAGE"]
    flux_radius = row["FLUX_RADIUS"]
    ellipticity = row["ELLIPTICITY"]

    width = scale * flux_radius * (1 + ellipticity)
    height = scale * flux_radius

    x_min = x_center - width / 2
    y_min = y_center - height / 2

    return pd.Series([x_min, y_min, width, height], index=["x", "y", "w", "h"])

def calculate_class(row, threshold=0.3):
    ellipticity = row["ELLIPTICITY"]
    return pd.Series([1 if ellipticity > threshold else 0], index=["class"])

def read_labels(labels_path: str) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_path, sep="\s+", names=dat_labels, comment="#")
    labels_df = labels_df[(labels_df["FLAGS"] == 0) & (labels_df["FLUX_RADIUS"] != 99.0)]

    labels_df[["x", "y", "w", "h"]] = labels_df.apply(calculate_bbox, axis=1)
    labels_df[["class"]] = labels_df.apply(calculate_class, axis=1)
    
    labels_df = labels_df[["x", "y", "w", "h", "class"]]
    return labels_df