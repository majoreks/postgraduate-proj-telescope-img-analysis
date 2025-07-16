import pandas as pd

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
DAT_FWHM_KEY = "FWHM_WORLD"
DAT_MAG_CALIB = "MAG_CALIB"
DAT_COMMENTS_KEY = "#"
DAT_LABELS = [
    DAT_X_KEY, DAT_Y_KEY, "ALPHA_J2000", "DELTA_J2000",
    "MAG_AUTO", "MAGERR_AUTO", DAT_FWHM_KEY, DAT_FLUX_KEY,
    DAT_ELLIPTICITY_KEY, "THETA_WORLD", "THETA_J2000", DAT_FLAGS_KEY, "MAG_CALIB","MAGERR_CALIB"
]

IMAGE_WIDTH, IMAGE_HEIGHT = 4096, 4108
# IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512

def calculate_bbox(row: pd.Series, pixel_scale: float) -> pd.Series:
    """
    Return bbox in COCO format [x_min, y_min, width, height]
    """
    
    x_center = row[DAT_X_KEY]
    y_center = row[DAT_Y_KEY]   
    fhwm_world = row[DAT_FWHM_KEY]
    mag_calib = row[DAT_MAG_CALIB]
    ellipticity = row[DAT_ELLIPTICITY_KEY]

    radius = fhwm_world / pixel_scale * 2
    if ellipticity > 0.5:
        radius = fhwm_world / pixel_scale * 4
    if mag_calib < 14:
        radius = fhwm_world / pixel_scale * 3

    x_min = x_center - radius
    y_min = y_center - radius

    width = radius * 2 
    height = radius * 2

    x_max = x_center + width / 2
    y_max = y_center + height / 2

    return pd.Series([x_min, y_min, x_max, y_max], index=COORDINATES_KEYS)

def calculate_class(_: pd.Series) -> pd.Series:
    return pd.Series([1], index=CLASSES_KEYS)

def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    valid_mask = (
        (df['x_min'] < df['x_max']) &
        (df['y_min'] < df['y_max']) &
        (df['x_min'] >= 0) & (df['x_max'] <= IMAGE_WIDTH) &
        (df['y_min'] >= 0) & (df['y_max'] <= IMAGE_HEIGHT)
    )

    df = df[valid_mask].reset_index(drop=True)

    df['width'] = df['x_max'] - df['x_min']
    df['length'] = df['y_max'] - df['y_min']
    df['size'] = df['width'] * df['length']

    Q1 = df['size'].quantile(0.25)
    Q3 = df['size'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - 1.5 * IQR)  
    upper_bound = Q3 + 1.5 * IQR

    return df[(df['size'] >= lower_bound) & (df['size'] <= upper_bound)]

def read_labels(labels_path: str, pixel_scale: float):
    labels_df = pd.read_csv(labels_path, sep=r"\s+", names=DAT_LABELS, comment=DAT_COMMENTS_KEY, header=None, engine="python")
    labels_df["PATH"] = labels_path

    if len(labels_df) == 0:
        return {
            "labels": labels_df,
            "reason": "initial_read"
        }
    
    labels_df = labels_df[(labels_df[DAT_FLAGS_KEY] == 0) & (labels_df[DAT_FLUX_KEY] != 99.0) & (labels_df[DAT_FLUX_KEY] > 0) & (labels_df[DAT_FLUX_KEY] < 50000)]

    if len(labels_df) == 0:
        return {
            "labels": labels_df,
            "reason": "initial_filter"
        }

    labels_df[COORDINATES_KEYS] = labels_df.apply(calculate_bbox, axis=1, pixel_scale=pixel_scale)
    labels_df[CLASSES_KEYS] = labels_df.apply(calculate_class, axis=1)

    if (labels_df[DAT_FLUX_KEY] < 0).any():
        print('negative flux radius found')
        return {
            "labels": labels_df,
            "reason": "negative_flux_radius"
        }
    # labels_df = filter_outliers(labels_df)
    # labels_df = labels_df[LABEL_KEYS]
    return {
            "labels": labels_df,
            "reason": "success"
        }