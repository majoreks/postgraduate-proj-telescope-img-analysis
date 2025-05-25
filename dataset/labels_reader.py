import pandas as pd

__all__ = ['read_labels']

X_KEY = "x_center"
Y_KEY = "y_center"
WIDTH_KEY = "width"
HEIGHT_KEY = "height"
CLASS_KEY = "class_id"
COORDINATES_KEYS = [X_KEY, Y_KEY, WIDTH_KEY, HEIGHT_KEY]
CLASSES_KEYS = [CLASS_KEY]
LABEL_KEYS = COORDINATES_KEYS + CLASSES_KEYS

DAT_X_KEY = "X_IMAGE"
DAT_Y_KEY = "Y_IMAGE"
DAT_FLUX_KEY = "FLUX_RADIUS"
DAT_ELLIPTICITY_KEY = "ELLIPTICITY"
DAT_FLAGS_KEY = "FLAGS"
DAT_COMMENTS_KEY = "#"
DAT_LABELS = [
    DAT_X_KEY, DAT_Y_KEY, "ALPHA_J2000", "DELTA_J2000",
    "MAG_AUTO", "MAGERR_AUTO", "FWHM_WORLD", DAT_FLUX_KEY,
    DAT_ELLIPTICITY_KEY, "THETA_WORLD", "THETA_J2000", DAT_FLAGS_KEY
]

def calculate_bbox(row: pd.Series, scale: float = 1.0) -> pd.Series:
    x_center = row[DAT_X_KEY]
    y_center = row[DAT_Y_KEY]
    flux_radius = row[DAT_FLUX_KEY]
    ellipticity = row[DAT_ELLIPTICITY_KEY]

    width = scale * flux_radius * (1 + ellipticity)
    height = scale * flux_radius

    return pd.Series([x_center, y_center, width, height], index=COORDINATES_KEYS)

def calculate_class(row: pd.Series, threshold: float = 0.3) -> pd.Series:
    ellipticity = row[DAT_ELLIPTICITY_KEY]
    return pd.Series([1 if ellipticity > threshold else 0], index=CLASSES_KEYS)

def read_labels(labels_path: str) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_path, sep='\s+', names=DAT_LABELS, comment=DAT_COMMENTS_KEY)
    labels_df = labels_df[(labels_df[DAT_FLAGS_KEY] == 0) & (labels_df[DAT_FLUX_KEY] != 99.0)]

    labels_df[COORDINATES_KEYS] = labels_df.apply(calculate_bbox, axis=1)
    labels_df[CLASSES_KEYS] = labels_df.apply(calculate_class, axis=1)

    labels_df = labels_df[LABEL_KEYS]
    return labels_df