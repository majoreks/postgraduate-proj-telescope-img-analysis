# Evaluation metrics
In this chapter, we introduce the key metrics that have been used to assess the performance of our object detection network. In object detection, metrics should focus on both the proper alignment of predicted bounding boxes relative to the ground truth and the correct classification of objects. The latter challenge is simplified in this project, as the problem has been reduced from a multi-class to a single-class scenario, defining only background and object-of-interest categories.

We utilize the `torchmetrics` package to ensure consistency and reproducibility, minimize development time, and delegate edge-case handling to well-tested implementations. This approach provides easy access to object detection and classification metrics, such as Intersection over Union (IoU), precision, and recall.

## Metrics rundown
Using `MeanAveragePrecision` and `intersection_over_union` the following metrics are calculated to evaluate performance of the model. 

### IoU related metrics
Intersection over Union (IoU) metrics quantify the degree of overlap between predicted and ground truth bounding boxes. IoU itself yields a quantitative measure of how closely the model’s predicted box matches the true object position. A larger IoU value reflects tighter overlap between the prediction and ground-truth box, indicating more accurate localization. The `intersection_over_union` function computes the IoU between each pair of predicted and ground truth boxes. In particular:
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

![alt text](media/1_Fh2VtPW6NNOvTPZ7ZWG3kQ.webp)


Using the matrix described above, we focus on the following 2 metrics

#### best_iou_per_gt

The `best_iou_per_gt` metric measures, for each ground-truth box, the highest IoU achieved by any predicted box and then averages these maxima across all ground-truths. Concretely, for each ground-truth in a batch we take the max over the IoU matrix’s columns (`iou.max(dim=0)`), concatenate these values, and compute the mean—yielding a single value that reflects how well the model covers true objects on average. This behavior is analogous to recall, since it indicates the degree to which real objects are met by at least one prediction, but it operates as a continuous, threshold-free measure of localization quality.

#### best_iou_per_prediction

The `best_iou_per_prediction` metric quantifies, for each predicted box, the highest IoU with any ground-truth box, then averages these values across all predictions. Practically, we take the max over the IoU matrix’s rows (`iou.max(dim=1)`), aggregate them over the batch, and compute the mean—providing insight into how precisely the model’s detections align with actual objects. This mirrors precision, as it reflects how many predictions are accurate, yet it remains a soft measure of overlap without requiring an IoU threshold to binarize true positives.

### Precision and recall related metrics

Mean Average Precision (mAP) and Mean Average Recall (mAR) extend IoU-based evaluation to capture both the trade-off between false positives and true positives and the model’s ability to find all objects. Using `torchmetrics.detection.mean_ap.MeanAveragePrecision`, predictions and ground-truths are aggregated to compute metrics at multiple IoU thresholds (by default COCO’s from 0.50 to 0.95 with step of 0.05). Mean Average Recall (mAR) is similarly derived by measuring recall at fixed numbers of detections per image (e.g., 1, 10, 100) across the same IoU thresholds and averaging it. Precision together with Recall describe the model’s effectiveness at producing correct positive detections and at identifying all true objects.

#### Precision
Precision is a key evaluation metric that measures the correctness of the model’s positive predictions by determining the proportion of true objects among all detected ones. It reflects the model’s effectiveness at filtering out false positives—higher precision implies the model makes confident, trustworthy detections with few incorrect alarms.

$$
precision = \frac{TP}{TP + FP}
$$

By using torchmetrics’ implementation of `MeanAveragePrecision`, we get mAP at different IoU thresholds, in particular at 50, 75, and the average of all thresholds, as well as for different object sizes, however, due to the dataset’s nature, where the overwhelming number of objects are very small, we mostly ignore those metrics.

#### Recall
Recall, also called sensitivity or the true positive rate, is a metric used for gauging a model’s performance, particularly in object detection. It quantifies the model’s ability to find every relevant object in an image, effectively measuring how comprehensive its detections are. A high recall score means the model succeeds at detecting the vast majority of true objects with few misses.

$$
recall = \frac{TP}{TP + FN}
$$

Similarly to precision, by using torchmetrics’ solution, we get mAR at different detection thresholds (defined by maximum number of detected objects). We define it as three equal steps from the maximum detections per image, where the maximum is predefined as a hyperparameter. Also, similarly to precision, we get recall at different object sizes, however, we ignore those metrics.


## Key metric
The key metric selected for this project has been mAP at 0.5 IoU threshold. This threshold ensures detected boxes match the objects of genuine scientific interest without imposing overly restrictive requirements, as it allows for minor localization variance while requiring meaningful overlap. By focusing on mAP at 0.5 threshold, we also reduce sensitivity to annotation inconsistencies described in the previous section so that our evaluation reflects true detection capability instead of artifacts arising from imperfect labels.

# References
1. https://medium.com/@henriquevedoveli/metrics-matter-a-deep-dive-into-object-detection-evaluation-ef01385ec62
2. https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
