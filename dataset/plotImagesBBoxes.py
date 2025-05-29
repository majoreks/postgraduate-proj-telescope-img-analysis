import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt

def plotFITSImageWithBoundingBoxes(image_data, labels_df):
    """
        PLOTS A FITS IMAGE WITH BOUNDING BOXES IN COCO FORMAT

    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    ax.imshow(image_data, cmap='gray', origin='lower',
            vmin=np.percentile(image_data, 5),
            vmax=np.percentile(image_data, 99))


    for row in labels_df:
        x = row[0]
        y = row[1]
        width = row[2]
        height = row[3]


        rect = patches.Rectangle(
            (x, y),
            width,height,
            linewidth=1,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

    ax.set_title("FITS Image with Bounding Boxes")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    #Should preserve the sky coordinates per each patch. I don't know if it's necessary right now, but for science it will definitely be interesting
    plt.show()
    pass