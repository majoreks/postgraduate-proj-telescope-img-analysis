import pandas as pd

__all__ = ['read_labels']

X_MIN_KEY = "x_min"
Y_MIN_KEY = "y_min"
X_MAX_KEY = "x_max"
Y_MAX_KEY = "y_max"
CLASS_KEY = "class_id"
COORDINATES_KEYS = [X_MIN_KEY, Y_MIN_KEY, X_MAX_KEY, Y_MAX_KEY]
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
    """
    Return bbox in COCO format [x_min, y_min, width, height]
    """
    
    x_center = row[DAT_X_KEY]
    y_center = row[DAT_Y_KEY]
    flux_radius = row[DAT_FLUX_KEY]
    ellipticity = row[DAT_ELLIPTICITY_KEY]

    scale = 5

    width = scale * flux_radius 
    height = scale * flux_radius* (1 + ellipticity)

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    return pd.Series([x_min, y_min, x_max, y_max], index=COORDINATES_KEYS)

def calculate_class(row: pd.Series, threshold: float = 0.3) -> pd.Series:
    ellipticity = row[DAT_ELLIPTICITY_KEY]
    return pd.Series([2 if ellipticity > threshold else 1], index=CLASSES_KEYS)

def read_labels(labels_path: str) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_path, sep=r'\s+', names=DAT_LABELS, comment=DAT_COMMENTS_KEY)
    labels_df = labels_df[(labels_df[DAT_FLAGS_KEY] == 0) & (labels_df[DAT_FLUX_KEY] != 99.0)]

    labels_df[COORDINATES_KEYS] = labels_df.apply(calculate_bbox, axis=1)
    labels_df[CLASSES_KEYS] = labels_df.apply(calculate_class, axis=1)

    labels_df = labels_df[LABEL_KEYS]
    return labels_df