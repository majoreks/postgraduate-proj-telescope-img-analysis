import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt

def voc_to_coco(boxes):
    return [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes.cpu().numpy()]

def plotFITSImageWithBoundingBoxes(image_data_, labels_ground_truth, labels_predictions, save_fig: bool = False, save_fig_sufix:str=None) -> None:
    """
        PLOTS A FITS IMAGE WITH BOUNDING BOXES IN COCO FORMAT

    """
    if labels_ground_truth is not None and len(labels_ground_truth['boxes']) > 0:
        labels_df = voc_to_coco(labels_ground_truth['boxes'])
        image_data = image_data_[0] if image_data_.ndim == 3 and image_data_.shape[0] == 1 else image_data_

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
        ax.imshow(image_data, cmap='gray', origin='lower',
            vmin=np.percentile(image_data, 5),
            vmax=np.percentile(image_data, 99), label='_nolegend_')


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

    if labels_predictions is not None and len(labels_predictions['boxes']) > 0:
        labels_df = voc_to_coco(labels_predictions['boxes'])
        
        for row in labels_df:
            x = row[0]
            y = row[1]
            width = row[2]
            height = row[3]


            rect = patches.Rectangle(
                (x, y),
                width, height,
                linewidth=1,
                edgecolor='green',
                facecolor='none'
            )
            ax.add_patch(rect)

    ax.set_title("FITS Image with Bounding Boxes")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    #Should preserve the sky coordinates per each patch. I don't know if it's necessary right now, but for science it will definitely be interesting
    if save_fig:
        if save_fig_sufix is not None and save_fig_sufix != "":
            filename = f'output/labels_{save_fig_sufix}.png'
        else:
            filename = 'output/labels.png'
        plt.savefig(filename, dpi=400)
    else:
        plt.show()
    
    plt.close(fig)