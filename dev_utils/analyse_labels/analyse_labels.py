from dev_utils.analyse_labels.labels_reader import read_labels
import matplotlib.pyplot as plt
import os
import pandas as pd

# root_dir = os.path.expanduser('/home/szymon/code/posgrado/postgraduate-proj-telescope-img-analysis/data_full')
root_dir = os.path.expanduser('~/data/posgrado-proj/images1000')
image_width, image_height = 4096, 4108
image_area = image_width * image_height

all_data = []

results = {}

def main():
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.dat') and '_imc_trl' in file:
                file_path = os.path.join(dirpath, file)
                try:
                    df = read_labels(file_path)

                    if df["reason"] != 'success':
                        if df["reason"] not in results:
                            results[df["reason"]] = {}
                            results[df["reason"]]["occurences"] = 0
                            results[df["reason"]]["paths"] = []
                        results[df["reason"]]["occurences"] += 1
                        results[df["reason"]]["paths"].append(file_path)
                        continue

                    all_data.append(df["labels"])
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    for key, value in results.items():
        print(f'{key} | {value["occurences"]}')

    if not all_data:
        print("No valid data files found.")
        exit()

    print(f'Num files found: {len(all_data)}')

    # df = read_labels('/home/szymon/code/posgrado/postgraduate-proj-telescope-img-analysis/data_full/2458950/CAT/TJO2458950.59431_V_imc_trl.dat')
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
    print(df.nlargest(10, 'size'))

    # Filter invalid bounding boxes
    valid_mask = (
        (df['x_min'] < df['x_max']) &
        (df['y_min'] < df['y_max']) &
        (df['x_min'] >= 0) & (df['x_max'] <= image_width) &
        (df['y_min'] >= 0) & (df['y_max'] <= image_height)
    )

    df_invalid = df[~valid_mask].reset_index(drop=True)
    df = df[valid_mask].reset_index(drop=True)

    print(f"Num objects after filtering: {len(df)}")
    print(f"Num invalid objects after filtering: {len(df_invalid)}")
    df_invalid.to_csv('output/dev/data/invalid.csv')
    print(f'Num unique files with invalid values {len(df_invalid["PATH"].unique())}')

    print('\n\n')

    # Detect outliers using IQR
    Q1 = df['size'].quantile(0.25)
    Q3 = df['size'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - 1.5 * IQR)  
    upper_bound = Q3 + 1.5 * IQR
    
    top_percent_size_bound = df['size'].quantile(0.995)
    top_percent_quantile_outlier = df[df['size'] >= top_percent_size_bound]
    print(f'num objects top 1%: {len(top_percent_quantile_outlier)}')
    print(f'Num unique files with top 1% outliers {len(top_percent_quantile_outlier["PATH"].unique())}')
    top_percent_quantile_non_outliers = df[df['size'] < top_percent_size_bound]

    # Identify outliers
    outliers = df[(df['size'] < lower_bound) | (df['size'] > upper_bound)]
    non_outliers = df[(df['size'] >= lower_bound) & (df['size'] <= upper_bound)]

    outliers.to_csv('output/dev/data/outliers.csv')
    print(f'Num unique files with outliers {len(outliers["PATH"].unique())}')

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
    plt.figure(figsize=(18, 5))

    plt.subplot(2, 3, 1)
    plt.hist(df['size'], bins=50, edgecolor='black', color='steelblue')
    plt.title('Histogram of Object Sizes (All Data)')
    plt.xlabel('Size (pixels²)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot: Filtered Histogram (excluding outliers)
    plt.subplot(2, 3, 2)
    plt.hist(non_outliers['size'], bins=50, edgecolor='black', color='seagreen')
    plt.title('Histogram of Object Sizes (No Outliers)')
    plt.xlabel('Size (pixels²)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot: Filtered Histogram (excluding outliers)
    plt.subplot(2, 3, 3)
    plt.hist(top_percent_quantile_non_outliers['size'], bins=50, edgecolor='black', color='seagreen')
    plt.title('Histogram of Object Sizes (No top 1% outliers)')
    plt.xlabel('Size (pixels²)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot: Filtered Histogram (excluding outliers)
    plt.subplot(2, 3, 4)
    plt.boxplot(df['size'])
    plt.title('Boxplot all data')
    plt.xlabel('Size (pixels²)')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(2, 3, 5)
    plt.boxplot(non_outliers['size'])
    plt.title('Boxplot no outliers')
    plt.xlabel('Size (pixels²)')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(2, 3, 6)
    plt.boxplot(outliers['size'])
    plt.title('Boxplot outliers')
    plt.xlabel('Size (pixels²)')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    directory = "output/dev/data/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory + 'labels_hist.png', dpi=400)

if __name__ == '__main__':
    main()