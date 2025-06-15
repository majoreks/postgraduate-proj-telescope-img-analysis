from dataset.labels_reader import read_labels
import matplotlib.pyplot as plt
import os
import pandas as pd

root_dir = 'data_full'
image_width, image_height = 4096, 4108
image_area = image_width * image_height

all_data = []

def main():
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.dat') and '_imc_trl' in file:
                file_path = os.path.join(dirpath, file)
                try:
                    df = read_labels(file_path)
                    all_data.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if not all_data:
        print("No valid data files found.")
        exit()

    print(f'Num files found: {len(all_data)}')

    # df = read_labels('/home/szymon/code/posgrado/postgraduate-proj-telescope-img-analysis/data_full/2458950/CAT/TJO2458950.59205_V_imc_trl.dat')
    df = pd.concat(all_data, ignore_index=True)

    print(f'Num objects found: {len(df)}')

    image_width = 4096
    image_height = 4108
    image_area = image_width * image_height

    # Calculate width, length, size, and ratio
    df['width'] = df['x_max'] - df['x_min']
    df['length'] = df['y_max'] - df['y_min']
    df['size'] = df['width'] * df['length']
    df['ratio'] = df['size'] / image_area

    # Compute averages
    avg_width = df['width'].mean()
    avg_length = df['length'].mean()
    avg_size = df['size'].mean()
    avg_ratio = df['ratio'].mean()
    max_width = df['width'].max()
    min_width = df['width'].min()
    max_length = df['length'].max()
    min_length = df['length'].min()

    # Print results
    print(f"Max Width: {max_width:.3f} pixels")
    print(f"Min Width: {min_width:.3f} pixels")
    print(f"Average Width: {avg_width:.3f} pixels")
    print(f"Average Length: {avg_length:.3f} pixels")
    print(f"Max Length: {max_length:.3f} pixels")
    print(f"Min Length: {min_length:.3f} pixels")
    print(f"Average Size: {avg_size:.2f} pixels²")
    print(f"Average Ratio to Image: {avg_ratio:.8f} ({avg_ratio * 100:.5f}%)")

    print("\nTop 10 Largest Objects by Size:")
    print(df.nlargest(10, 'size')[['x_min', 'y_min', 'x_max', 'y_max', 'class_id', 'size']])

    # Filter invalid bounding boxes
    valid_mask = (
        (df['x_min'] < df['x_max']) &
        (df['y_min'] < df['y_max']) &
        (df['x_min'] >= 0) & (df['x_max'] <= image_width) &
        (df['y_min'] >= 0) & (df['y_max'] <= image_height)
    )

    df = df[valid_mask].reset_index(drop=True)

    print(f"Num objects after filtering: {len(df)}")

    print('\n\n')

    # Detect outliers using IQR
    Q1 = df['size'].quantile(0.25)
    Q3 = df['size'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - 1.5 * IQR)  
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df['size'] < lower_bound) | (df['size'] > upper_bound)]
    non_outliers = df[(df['size'] >= lower_bound) & (df['size'] <= upper_bound)]

    # Compute averages
    avg_width = non_outliers['width'].mean()
    avg_length = non_outliers['length'].mean()
    avg_size = non_outliers['size'].mean()
    avg_ratio = non_outliers['ratio'].mean()
    max_width = non_outliers['width'].max()
    min_width = non_outliers['width'].min()
    max_length = non_outliers['length'].max()
    min_length = non_outliers['length'].min()

    print('\n\n')

    # Print results
    print(f"non_outliers Max Width: {max_width:.3f} pixels")
    print(f"non_outliers Min Width: {min_width:.3f} pixels")
    print(f"non_outliers Average Width: {avg_width:.3f} pixels")
    print(f"non_outliers Average Length: {avg_length:.3f} pixels")
    print(f"non_outliers Max Length: {max_length:.3f} pixels")
    print(f"non_outliers Min Length: {min_length:.3f} pixels")
    print(f"non_outliers Average Size: {avg_size:.2f} pixels²")
    print(f"non_outliers Average Ratio to Image: {avg_ratio:.8f} ({avg_ratio * 100:.5f}%)")

    print(f"\nOutlier Thresholds:")
    print(f" - Lower bound: {lower_bound:.2f}")
    print(f" - Upper bound: {upper_bound:.2f}")
    print(f"\nDetected {len(outliers)} outlier(s) based on size:")

    print("\nnon_outliers Top 10 Largest Objects by Size:")
    print(non_outliers.nlargest(10, 'size')[['x_min', 'y_min', 'x_max', 'y_max', 'class_id', 'size']])

    # Plot: Full Histogram
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(df['size'], bins=50, edgecolor='black', color='steelblue')
    plt.title('Histogram of Object Sizes (All Data)')
    plt.xlabel('Size (pixels²)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot: Filtered Histogram (excluding outliers)
    plt.subplot(1, 2, 2)
    plt.hist(non_outliers['size'], bins=50, edgecolor='black', color='seagreen')
    plt.title('Histogram of Object Sizes (No Outliers)')
    plt.xlabel('Size (pixels²)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    directory = "output/dev/data/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory + 'labels_hist.png', dpi=400)

if __name__ == '__main__':
    main()