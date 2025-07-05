import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def voc_to_coco(boxes):
    return [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes.cpu().numpy()]

def plotFITSImageWithBoundingBoxes(
    image_data_,
    labels_ground_truth,
    plot_scores: bool = False,
    labels_predictions = None,
    output_path='output',
    save_fig: bool = False,
    save_fig_sufix: str = None,
    title_sufix: str = None
) -> None:
    """
    Plots a FITS image with COCO-format bounding boxes and shows class IDs in a separate legend.
    - GT boxes: red-orange tones
    - Prediction boxes: green-yellow tones
    """
    image_data = image_data_[0] if image_data_.ndim == 3 and image_data_.shape[0] == 1 else image_data_
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(
        image_data,
        cmap='gray',
        origin='lower',
        vmin=np.percentile(image_data, 5),
        vmax=np.percentile(image_data, 99),
    )

    # Paletas de colores
    gt_colors = ['red', 'darkorange', 'orangered', 'tomato', 'salmon']
    pred_colors = ['green', 'yellowgreen', 'lime', 'gold', 'yellow']

    legend_elements = {}

    # === Ground Truth ===
    if labels_ground_truth is not None and len(labels_ground_truth['boxes']) > 0:
        boxes_gt = voc_to_coco(labels_ground_truth['boxes'])
        class_gt = labels_ground_truth['labels'].cpu().numpy()

        for row, class_id in zip(boxes_gt, class_gt):
            x, y, width, height = row
            color = gt_colors[class_id % len(gt_colors)]
            label = f"GT Class {class_id}"

            rect = patches.Rectangle(
                (x, y), width, height,
                linewidth=1.5,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)

            # Añadir al diccionario de leyenda
            if label not in legend_elements:
                legend_elements[label] = patches.Patch(edgecolor=color, facecolor='none', label=label, linewidth=1.5)

    # === Predicciones ===
    if labels_predictions is not None and len(labels_predictions['boxes']) > 0:
        boxes_pred = voc_to_coco(labels_predictions['boxes'])
        class_pred = labels_predictions['labels'].cpu().numpy()

        scores_pred = None
        if plot_scores and 'scores' in labels_predictions:
            scores_pred = labels_predictions['scores'].cpu().numpy()

        for idx, (row, class_id) in enumerate(zip(boxes_pred, class_pred)):
            x, y, width, height = row
            color = pred_colors[class_id % len(pred_colors)]
            label = f"Pred Class {class_id}"

            rect = patches.Rectangle(
                (x, y), width, height,
                linewidth=1.5,
                edgecolor=color,
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)

            if scores_pred is not None:
                score = scores_pred[idx]

                text_x = x - width / 2
                text_y = y + height + 2
                ax.text(
                    text_x, text_y - 2,
                    f"{score:.2f}",
                    fontsize=8,
                    color=color,
                    va='bottom',
                )

            if label not in legend_elements:
                legend_elements[label] = patches.Patch(edgecolor=color, facecolor='none', label=label, linestyle='--', linewidth=1.5)

    ax.set_title("FITS Image with Bounding Boxes" + (f" - {title_sufix}" if title_sufix else ""))
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    # Mostrar leyenda fuera del plot
    if legend_elements:
        ax.legend(
            handles=list(legend_elements.values()),
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            fontsize=9,
            frameon=False
        )

    plt.tight_layout()

    # Guardar o mostrar
    if save_fig:
        filename = f"{output_path}/labels_{save_fig_sufix}.png" if save_fig_sufix else f"{output_path}/labels.png"
        plt.savefig(filename, dpi=400, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)