# Evaluation metrics
In this chapter, we introduce the key metrics that have been used to assess the performance of our object detection network. In object detection, metrics should focus on both the proper alignment of predicted bounding boxes relative to the ground truth and the correct classification of objects. The latter challenge is simplified in this project, as the problem has been reduced from a multi-class to a single-class scenario, defining only background and object-of-interest categories.

We utilize the `torchmetrics` package to ensure consistency and reproducibility, minimize development time, and delegate edge-case handling to well-tested implementations. This approach provides easy access to object detection and classification metrics, such as Intersection over Union (IoU), precision, and recall.

## Metrics rundown
Using `MeanAveragePrecision` and `intersection_over_union` the following metrics are calculated to evaluate performance of the model. 

### IoU related metrics
Intersection over Union (IoU) metrics quantify the overlap between predicted and ground truth bounding boxes, serving as the cornerstone for localization accuracy. The `intersection_over_union` function computes the IoU between each pair of predicted and ground truth boxes. In particular:
```py
intersection_over_union(pred["boxes"], target["boxes"], aggregate=False)
```
returns `N x M` matrix of IoU scores where `N` is the number of predicted boxes and `M` is the number of target boxes such that each entry (i, j) is   
$$
\mathrm{IoU_{ij}} = \frac{area(\;pred_{\text{i}}\;\cap\; gt_{\text{j}})}{area(\;pred_{\text{i}} \;\cup\; gt_{\text{j}})}
$$

where

- $pred_{\mathrm{i}}$ is the $i$-th predicted bounding box
- $gt_{\mathrm{j}}$ is the $j$-th ground-truth bounding box
- $\cap$ denotes the geometric intersection of the two boxes
- $\cup$ denotes their union
- $area()$ indicates the area of the region

Using the matrix described above, we focus on the following 2 metrics

#### best_iou_per_gt

#### best_iou_per_prediction

### Precision and recall related metrics


# References
1. https://medium.com/@henriquevedoveli/metrics-matter-a-deep-dive-into-object-detection-evaluation-ef01385ec62
