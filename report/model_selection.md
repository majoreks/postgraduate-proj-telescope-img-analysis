# Model selection
In this chapter, we describe our approach to identifying an optimal detection architecture for a dataset dominated by numerous small objects. Our entry point was the two-stage Faster R-CNN framework, selected for its modularity and strong track record in general-purpose object detection. From this baseline, we conducted a series of controlled experiments to understand how architectural and training choices impact performance on our problem, inference speed, model complexity and training time. Key factors explored include:
- Backbone variants: We compared different ResNets to assess if smaller backbone networks would be sufficient for our relatively simple images.
- We evaluated both the original Faster R-CNN (“v1”) _[1]_ and its refined “v2” variant _[2][3]_.
- Pretraining and transfer learning: All backbones have the possibility of initializing with ImageNet-pretrained weights, then fine-tuning on our target data if needed.
- Layer freezing strategies: We experimented with different approach to which layers should be frozen and unfrozen to see how it would affect model's performance and training speed.

## Preliminary experiments
### Layer freezing
### Model architecture

# References
1. Ren, S., He, K., Girshick, R., & Sun, J. (2016). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. arXiv preprint areXiv:1506.01497. https://arxiv.org/abs/1506.01497
2. PyTorch. (n.d.). torchvision.models.detection.fasterrcnn_resnet50_fpn_v2 [Documentation]. Torchvision main documentation. Retrieved July 13, 2025, from https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html
3. Li, Y., Xie, S., Chen, X., Dollar, P., He, K., & Girshick, R. (2021). Benchmarking Detection Transfer Learning with Vision Transformers. arXiv preprint arXiv:2111.11429. https://arxiv.org/abs/2111.11429