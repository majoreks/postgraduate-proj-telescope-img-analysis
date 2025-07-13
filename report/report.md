# SAT PROJECT

[**1 INTRODUCTION	2**](#1-introduction)

[1.1 Motivation: PROBLEM TO SOLVE	2](#1.1-motivation:-problem-to-solve)

[2 JOAN ORÓ TELESCOPE DATASET	2](#2-joan-oró-telescope-dataset)

[2.1 Image characteristics	3](#2.1-image-characteristics)

[2.2 Ground truth characteristics	4](#2.2-ground-truth-characteristics)

[2.3 Dataset Preprocessing Procedure	6](#2.3-dataset-preprocessing-procedure)

[2.3.1 Manual cleaning and curation of the dataset	6](#2.3.1-manual-cleaning-and-curation-of-the-dataset)

[2.3.2 Image selection, splitting, and cropping	10](#2.3.2-image-selection,-splitting,-and-cropping)

[**3\. SYSTEM ARCHITECTURE	10**](#4.-system-architecture)

[3.1 Model selection	10](#4.1-model-selection)

[3.2 Adapting RGB pre-trained Models for Monochromatic inputs	13](#4.2-adapting-rgb-pre-trained-models-for-monochromatic-inputs)

[3.3 Fine tuning strategies for specific object detection	13](#4.3-fine-tuning-strategies-for-specific-object-detection)

[3.3 Data augmentation	13](#4.4-data-augmentation)

[3.3 Metrics and loss criterion	13](#heading=h.xdqeqmhe54xo)

[3.4 Checkpoints and Early Stopping	13](#4.6-checkpoints-and-early-stopping)

[3.1 Main Hyperparameters	14](#4.7-main-hyperparameters)

[3.5 Working Environment	14](#heading=h.8ha8vao109aq)

[**4\. Variations and experiments	14**](#5.-experiments)

[4.3 Hyperparameter search experiments	14](#5.2-hyperparameter-search-experiment)

[**4\. Results	17**](#6.-results)

[4.1 Model comparison	17](#6.1-model-comparison)

[Best performance	17](#best-performance)

[Experiments that have been run	17](#heading=h.uomqm1h8tdzf)

[4.2 Effect of Non Maximum Suppression Threshold on Object Detection	17](#6.2-effect-of-non-maximum-suppression-threshold-on-object-detection)

[4\. Modifications implemented to the models / Special Techniques	18](#4.-modifications-implemented-to-the-models-/-special-techniques)

[5\. Inference and Validation	18](#5.-inference-and-validation)

[**CONCLUSIONS	19**](#conclusions)

[**Future improvements	19**](#future-improvements)

[Index	19](#index)

[**Bibliography	22**](#bibliography)

# 

# 1 INTRODUCTION {#1-introduction}

## 1.1 Motivation: PROBLEM TO SOLVE  {#1.1-motivation:-problem-to-solve}

We have used Artificial Intelligence and DL algorithms to solve the problem of **automatic detection of relevant astrophysical objects (such as stars, galaxies, or cosmic structures)** in telescope images from Joan Oró in the Montsec Observatory \[ref 1\] — a task that is **highly challenging due to the variability, noise, and complexity of the data**.

Traditional methods or rule-based systems fail to generalize across different kinds of images, especially when faced with:

* Instrumental noise (e.g., traces, bright spots, edge artifacts),  
* Inconsistent annotations or missing metadata,  
* Overlapping or barely visible objects.

Using **deep learning object detection models** (Faster R-CNN with ResNet backbones), we developed and trained a system capable of:

* Learning complex visual patterns from astronomical data,  
* Generalizing detection across varied image conditions,  
* Evaluating predictions using robust metrics (mAP, IoU),  
* Automatically adjusting to different detection thresholds (via NMS and hyperparameter tuning).

# 2 JOAN ORÓ TELESCOPE DATASET {#2-joan-oró-telescope-dataset}

The Team has worked with a set of **astronomical images**, both **raw** and **pre-processed**, with the goal of detecting **relevant astrophysical objects**, such as **stars**, **galaxies**, or other celestial bodies.

The dataset has two big blocks: the *.fits* images and the ground truth, which is a file with *.dat* extension. As it is stated in the bibliography \[ref 2\], fits images are very heavy to manage in ram memory during execution.

## 2.1 Image characteristics {#2.1-image-characteristics}

* **Duplicated images** that were not properly detected by filename but could be identified through content or metadata.  
* **High variability** in the appearance of these objects: they can differ greatly in **shape**, **brightness**, **size**, and **contrast**.

* The presence of **shiny artifacts**, **diffraction patterns**, and **instrumental noise**, which can mislead detection models.

* The complexity of differentiating between background noise and actual relevant objects.

* Some images are particularly challenging — we refer to them as the "**difficult ones**" — where standard detection techniques fail or produce unreliable results.

Examples of defects and **variability** of the data:

| A satellite can be detected | Depending on the readout, different regions of the detector can have different biases | Example of fringes due to illumination in the background or filter defect | Examples of fringes II |
| ----- | ----- | ----- | ----- |
| ![][image1] | ![][image2] | ![][image3] | ![][image4] |

Different illumination patterns and “contrast” between images due to different exposition time, different filters and different background illumination (sky luminosity counts). If the moon is near, or the object is near the twilight it can affect the background counts.

| Saturated objects may appear | Gradient in the background  illumination | A trace from a comet | A galaxy with high ellipticity |
| :---- | :---- | :---- | :---- |
| ![][image5] | ![][image6] | ![][image7] | ![][image8] |

| A big galaxy with high elipticity and different size | Different background and appearance of the object due to the use of two different filters |
| ----- | ----- |
| ![][image9] | ![][image10] |

## 2.2 Ground truth characteristics {#2.2-ground-truth-characteristics}

After the observation, images are processed with a professional pipeline in the telescope and every image acquires (or should be paired to) its catalogue, a .dat file that is saved with the image. 

![][image11]

This catalogue has astronomical information about the observed objects such as luminosity, position in the sky, ellipticity of the objects, etc. As other telescopes over the world, it follows the sextractor format \[ref 3\]. However, there are some issues regarding this file:

**Incorrect or incomplete annotations**, including:

1. Images without catalogs, or with empty or inconsistent catalogs.  
2. Poorly defined bounding boxes, especially for medium and large objects.  
3. **Mismanaged or missing metadata**, which prevented automation of tasks such as duplicate detection and **labels with semantic or spatial errors**, which negatively impacted training and metric evaluation.

| Example of issue 2\) | Example of issue 2\) | Example of issue 3\) |
| :---- | :---- | :---- |
| ![][image12] | ![][image13] | ![][image14] |

## 2.3 Dataset Preprocessing Procedure {#2.3-dataset-preprocessing-procedure}

### 2.3.1 Manual cleaning and curation of the dataset {#2.3.1-manual-cleaning-and-curation-of-the-dataset}

Taking into account the previous issues a pre-processing of the data was done a manual clearing and curation of the dataset.

* Individual review of hundreds of images to label the images.  
* Identification of duplicates using metadata and visual content.  
* Correction or removal of faulty bounding boxes.  
* Skip objects that saturate the detector and cannot be considered ***relevant objects*** or ***scientific objects,*** which will be labelled as ESO (example of saturated object).

Examples of the labels after the 1 by 1 inspection:

| ESO, example of a saturated object | Profiles of the normalized flux |
| ----- | :---- |
| ![][image15] | ![][image16]![][image17] |

| ESO, COMET | TRACE | GALAXY | ESO |
| :---- | :---- | :---- | ----- |
| ![][image18] | ![][image19] | ![][image20] | ![][image21] |
| ![][image22] | ![][image23] | ![][image24] | ![][image25] |

| BINARY | LIGHT |
| :---- | :---- |
| ![][image26] | ![][image27] |

This processing gives as a result a generation of a **clean, high-quality dataset** with [labels associated to the file](https://drive.google.com/drive/folders/1bJ2Juznn3Gzg7jkGEJmCCFB1bcMXFdFk?usp=drive_link), in order to filter datasets and being able to generate controlled and labelled ground truth to perform the experiments:

![][image28]

In the following graph, the variance of the images is summarized ([script](https://colab.research.google.com/drive/1Zildsp_DplN5QcJmf9-2w8AyHPB6686q?usp=drive_link)):

![][image29]

To avoid dealing with variability of the objects, [some datasets were prepared](https://colab.research.google.com/drive/1r49XJp1SminHT5qjxWdTCSZ7A8Eq813h?usp=drive_link), selecting the labels OK, OK and ESO.

The model of this project is trained with a [dataset of 299 images](https://colab.research.google.com/drive/1r49XJp1SminHT5qjxWdTCSZ7A8Eq813h?usp=drive_link) with reliable labels such as this image and to generate it we have filtered the images with no saturated objects and no weird features excluding galaxies, comets, or binary systems to make sure that the model learn how to differentiate a star from the background, and so, is capable to detect **relevant objects**.

![][image30]

Even though the dataset has been reduced to retain only reliable labels, there are still plenty of extreme outliers that heavily skew the data. The diagrams below show histograms and boxplots of object sizes with and without outliers
![alt text](media/data_object-distribution-hist.png)
![alt text](media/data_object-distribution-boxplot.png)
In both cases it can be seen that there are extreme outliers heavily skewing the data beyond it’s otherwise normal distribution.


Different IA models with the MODELS 

Methodology 

DATA preparation. → CRIS  
GROUND TRUTH → repetition, decisions taken, reduced labelled dataset.   
…. This is the data that we have to train.  
	Talk abou the croppings (train, test, validation)

→ not technical, natural language as possible, puting some examples  
Problemm references, not using deep learning, selling the problem, how difficult it is  
Points not relevant. Saturated objects.

* **Ground Truth (GT) improvement**

  * Manually cleaned GT annotations

  * Filtered void, incomplete, and duplicate entries

  * Verified catalog presence and object count per image

### 2.3.2 Image selection, splitting, and cropping {#2.3.2-image-selection,-splitting,-and-cropping}

In case any image is found without ground truth, it is set apart from the training dataset, so they still can be used for inference. Right after, the images are hard splitted as 81% train, 9% validation, and 10% test.

In the cleaned dataset the average size of an object is 319.46 pixels² which gives an average ratio of object to the image of 0.00001899 with around 345 objects per image on average. 
In order to reduce computation resources, as well as reduce the number of objects per image and increase the relative size of object to the image, the images are split into 512x512 pixel images (configurable), and information on the cropping coordinates is added to the product metadata. A “cropped” version of the ground truth is also generated including only the entries in the cropped region, removing any position offset. If no entries are available at a given crop, neither the cropped image or the cropped ground truth are saved.

After this operation the average size of an object stays nearly the same at 319.48 pixels² (difference most likely coming from edge cases handling), however the average ratio of an object to an image increases to 0.00121871 and the average number of objects per image drops down to around 48.

# 3\. Working environment

The project was developed on local machines. Training and experiments were done using both local consumer machine with GPU (RTX 4070 Ti), Google Colab and Google Compute Engine to be able to use more powerful machines and perform different experiments in parallel.

# 4\. System architecture {#4.-system-architecture}

## 4.1 Model selection {#4.1-model-selection}

The goal is to explore how modern deep learning techniques could be applied to astronomical analysis. Given that the available data consists primarily of 2D monochromatic images from telescopes, it was natural to focus on convolutional neural networks and other architectures designed to process visual inputs. To guide this selection, we reviewed both general-purpose object detection architectures and domain-specific models reported in the literature. A large part of this benchmarking effort was informed by the comprehensive literature review presented in Radio Astronomical Images Object Detection and Segmentation: A Benchmark on Deep Learning Methods \[3\], as well as insights drawn from individual studies such as Mask Galaxy: Morphological Segmentation of Galaxies \[4\]. The benchmarking process grouped models into the following main categories:

* Two-stage detectors: Two-stage models first generate candidate regions where objects might be, and then classify each region and refine its position. Examples include:  
  * Fast R-CNN, which uses a CNN backbone (e.g., VGG16) to extract features from the entire image. Then, it applies Region of Interest (ROI) pooling to extract features from proposed object regions. However, these proposals come from an external algorithm, usually Selective Search, which is not part of the neural network and doesn’t learn from the data.   
  * Faster R-CNN removes this bottleneck by introducing a Region Proposal Network (RPN), a lightweight CNN that slides over the feature map and proposes regions likely to contain objects. This makes the model end-to-end trainable. Additionally, it often uses a Feature Pyramid Network (FPN) as a backbone. FPNs build multi-scale feature maps, allowing the network to detect objects of different sizes, a key benefit for astronomy, where stars and galaxies can appear at vastly different scales.  
  * Mask R-CNN builds directly on Faster R-CNN. It adds a third output branch to the architecture: a segmentation, enabling pixel-level instance segmentation. It also improves how features are extracted from regions using RoIAlign, a refinement over ROI that helps preserving spatial accuracy, an important improvement for scientific images where precision matters.  
* One-stage detectors:  One-stage models detect and classify objects in a single pass, without a separate region proposal step, making them faster but sometimes less precise than two-stage models. In these architectures, the entire image is processed at once,  
  * YOLO (You Only Look Once): Divides the image into a grid and lets each grid cell predict bounding boxes and class probabilities. It’s extremely but may struggle with very small or overlapping objects.  
  * Single Shot MultiBox Detector (SSD): Improves on YOLO by using multiple feature maps at different resolutions, allowing better detection of both small and large objects.  
  * RetinaNet: Builds on SSD with a stronger backbone and introduces focal loss, a technique that reduces the influence of easy background examples and focuses the training on harder, less frequent objects, particularly useful for datasets with high class imbalance, like astronomical images where most pixels are background.  
  * EfficientDet: Prioritizes model efficiency by using a bi-directional Feature Pyramid Network (BiFPN), which allows the model to combine information from both shallow and deep layers. Additionally, it uses compound scaling, a technique that jointly scales the model’s depth, width, and input resolution in a balanced manner.   
* Domain-specific models. Several models have been specifically designed or adapted for astronomical image analysis  
  * AstroYOLO: Adaptation of YOLO for astronomical surveys.  
  * PSDetNet, which is designed for the detection of point sources — such as stars — in particularly noisy environments, which are common in deep-sky or radio observations.  
  * PI-AstroDeconv is a model designed to detect very faint or blurred objects in astronomical images. It integrates PSF deconvolution, a process that corrects image distortion caused by telescopes or the atmosphere, directly into the network.   
  * Mask R-CNN (galaxy-focused) has been successfully applied to tasks such as galaxy segmentation and morphological classification.  
* Transformer-based detectors: Transformer-based models, originally models developed for natural language processing, have ability to capture global relationships in an image, meaning they can understand how different parts of an image relate to each other, even if they are far apart.  These models show promising results, particularly in complex spatial contexts. However, the high computational cost and training complexity make them more suitable for future work than for current baseline evaluations. Some examples are:   
  * DETR (Detection Transformer) combines a CNN backbone (for extracting visual features) with a transformer encoder-decoder, allowing the model to directly predict object positions and classes without needing anchor boxes or region proposals. It’s particularly effective at detecting overlapping or irregular objects but it requires significant computational resources. STAR-DETR is a specialized version of DETR, optimized for detecting space-related targets, such as satellites or objects in low-Earth orbit.  
  * RelationNet adds a transformer-like module between the feature extractor and the prediction layers, allowing the model to "look at" multiple regions at once and learn how they influence each other.

After evaluating a wide range of models, Faster R-CNN was selected as the architecture for this project. This choice was guided primarily by academic considerations, as this model had been studied and implemented in the context of our coursework, which makes this model a pedagogically sound option. Models based on transformers and one-stage detectors were not selected due to task-specific constraints. Transformer-based models come with a high computational cost, leading in high training times to perform well. One-stage detectors offer high inference speed and are suitable for real-time applications, but this is not a critical requirement in our astronomical task, where accuracy is prioritized over speed.   
We acknowledge that in a professional setting, the choice of Faster R-CNN would benefit from further refinement. Several models specifically designed for astronomical applications, such as AstroYOLO, PSDetNet, or Pi-AstroDeconv (as explained before) have shown promising performance in recent literature. In this sense, the selection of Faster R-CNN should be viewed as a solid academic baseline rather than an optimal solution.   
That said, Faster R-CNN is a robust and flexible architecture and it has also been successfully adopted in published astronomical research. For instance, CLARAN (Wu et al., 2018\) \[1\] applies a Faster R-CNN variant to classify complex radio morphologies, while Burke et al. (2019) \[2\] use a Mask R-CNN (built on Faster R-CNN) for deblending and classifying blended sources in deep-sky surveys. These examples demonstrate that, despite being a general-purpose model, Faster R-CNN remains competitive choice for object detection and segmentation in astronomy

Maximum number of objects detected increased\!

## 4.2 Adapting RGB pre-trained Models for Monochromatic inputs {#4.2-adapting-rgb-pre-trained-models-for-monochromatic-inputs}

Object detection models are generally trained with general purpose RGB images. However, the telescope images are monochromatic and might not even be in the Red, Green or Blue channels. Several techniques can be applied to adapt a RGB network to a monochromatic image:

* Replicating 3 times the original image to have a 3 channel image.  
* Modify the input CNN to have 1 input and use as weights the average of the original 3-channel input CNN.  
* Modify the input CNN to have 1 input, randomly initialize the weights, and train it from scratch during fine tuning. 

In

## 4.3 Fine tuning strategies for specific object detection {#4.3-fine-tuning-strategies-for-specific-object-detection}

Pretrained object detection networks are trained with a wide variety of objects. Their use for specific categories of objects (such as celestial bodies) require fine tuning, even more if the goal is to differentiate between objects that are so similar (the difference between a galaxy and a star is small compared to the difference of a cat and a car).

Reducing the number of layers might seem a logical decision since the objects have low complexity (white dots or ellipsoids in an almost black background), so the network becomes lighter and faster to train. However, the difference between a star and a galaxy might be encoded in deeper and more abstract features \[Min, 2022\], and therefore it might be a counterproductive decision.

Fine tuning strategies might include:

* Selective Parameter-Efficient Fine-Tuning, or partial fine tuning, which trains only part of the layers of the model, typically the latest ones, freezing the backbone. A variation includes the Dynamic Backbone Freezing, which freezes and unfreezes alternately the backbone during the training stage. This technique allows to preserve low-level generic features and to include new specific features.  
* Additive Parameter-Efficient Fine-Tuning, which introduces bottleneck layers in the pretrained model, and only these layers are trained.  
* Reparametrization using techniques such as Low-Rank Adaptation, that allow to represent the current parameters into a lower dimensional form so to find which parameters need to be retuned, reducing the number of parameters to be retuned up to 99% \[Zhang, 2025\].  
  


## 4.4 Data augmentation {#4.4-data-augmentation}

Data augmentation is conducted using the Albumentations library, which allows to crop, rotate, zoom, etc. not only images, but also bounding boxes and masks. The library can also discard invalid bounding boxes (those out of a cropping, for instance).

For data augmentation, originally images were cropped to get 1 image of 512x512 pixels ensuring at least one bounding box,, and  then randomly rotated from 0º to 270º in steps of 90º. After many images were discarded from manual filtering, the images were hard cropped into subimages of 512x512 to overcome the drastic decrease of useful data.

## 4.5 Non Maximum Suppression Threshold adjustment

The Non-Maximum Suppresion (NMS) algorithm threshold is responsible for reducing the number of regions of interest proposed by the region proposal network. The algorithm is based on the Intersection over Union metric (IoU) as the intersection of two bounding boxes divided by their union. If the value is over the given threshold (0.5 by default), the two boxes are considered to be covering the same object, and the smaller one is chosen. The threshold can be changed by modifying the variable model.roi\_heads.nms\_thresh. If objects tend to be too close, the NMS algorithm might propose just one region for the two objects. Furthermore, experiments in this project showed that noisier images tend to show more proposed ROIs over the same object, so the NMS threshold should be adjusted.

## 4.6 Checkpoints and Early Stopping {#4.6-checkpoints-and-early-stopping}

* **Checkpoints & Early Stopping**

  * Implemented checkpoint saving

  * Considered early stopping (optional, based on wandb config)  
  * 

## 4.7 Main Hyperparameters {#4.7-main-hyperparameters}

* `batch_size`: \[4, 8, 16\]

  * `learning_rate`: log-uniform \[1e-6 to 1e-2\]

    * `nms_threshold`: \[0.3, 0.5, 0.7\]

    * `early_stopping_patience`: \[0, 3, 5\]

    * `weight_decay`: \[1e-5, 1e-4, 1e-2\]

# 5\. EXPERIMENTS {#5.-experiments}

## 4.1 Planned experiments

Explain here the context of all the experiments planned and performed

| Model  | Trained weights  | Backbone architecture | Modification  |  |
| :---- | :---- | :---- | :---- | :---- |
| Faster R-CNN v1 |  **ResNet backbones** on COCO dataset | ResNet-18 ResNet-34 ResNet-50 ResNet-101 ResNet-152 ¿? | NMS Cropping Checkpoints Early stopping |  |
| Faster R-CNN v2 |  |  |  |  |

## 5.2 Hyperparameter search experiment {#5.2-hyperparameter-search-experiment}

Weights & Biases (W\&B) provides several methods to perform systematic hyperparameter search, each with its own underlying mechanism. Since we already use W\&B to track our model training, metrics, and artifacts, it was a natural choice to also leverage its integrated *sweeps* functionality to automate and manage our hyperparameter experiments. 

Setting up a sweep in W\&B requires defining three main components: (1) the **objective metric** to optimize , (2) the **hyperparameter space**, specifying possible values or distributions for each parameter, and (3) the **search strategy**, such as grid, random, or Bayesian. Once defined, the process consists of two main steps:

* **First**, the sweep is registered using `wandb.sweep()`, which takes the configuration dictionary (or YAML) and returns a unique `sweep_id` identifying the experiment.  
* **Second**, the sweep is executed via `wandb.agent()`, which continuously samples new configurations based on the chosen search strategy and launches training runs accordingly.

In the following sections, we provide more details on the three key components mentioned above—(1) the optimization metric, (2) the hyperparameter space, and (3) the search strategy—as applied to our specific experimental setup.

**Optimization metric and hyperparameter space**

In our experiments, the objective metric selected for optimization was mean Average Precision at IoU \= 0.5 (`map_50`). This metric is standard in object detection tasks and captures both classification accuracy and spatial alignment between predicted and ground-truth bounding boxes. We chose `map_50` specifically because it provides a balanced signal in the presence of fuzzy or ambiguous object boundaries, which are common in astronomical imagery.

The hyperparameter space was designed to explore a range of values that are known to influence both model convergence and generalization. It includes:

* `batch_size`: values of 4, 8, and 16  
  These values help explore the trade-off between training stability, memory efficiency, and gradient estimation quality.  
* `learning_rate`: sampled log-uniformly between 1e-6 and 1e-2  
  This range allows the sweep to test both conservative and aggressive learning regimes, capturing several orders of magnitude of possible behavior.  
* `weight_decay`: values of 1e-5, 1e-4, and 1e-2  
  This regularization parameter helps prevent overfitting, especially in small or imbalanced datasets typical in scientific domains.  
* `early_stopping_patience`: values of 0, 3, and 5  
  This parameter controls how many epochs without improvement are tolerated before stopping. Exploring different values allows us to assess the sensitivity of training time and convergence to early-stopping aggressiveness.

**Search strategies**

W\&B sweeps support several **hyperparameter search strategies**, allowing users to choose how parameter combinations are selected and evaluated during experimentation. Below we summarize the most common search strategies supported by W\&B.

* **Grid Search** is an exhaustive strategy that evaluates all possible combinations of hyperparameter values defined in the search space. It systematically iterates through each possible configuration.  
* **Random Search** selects combinations of hyperparameters at random from the specified distributions. It does not attempt to cover the entire space uniformly, but often finds good solutions with fewer evaluations.  
* **Bayesian Optimization** builds a probabilistic model of the objective function (typically using Gaussian Processes or Tree-structured Parzen Estimators). It uses this surrogate model to predict which regions of the hyperparameter space are likely to yield better results, balancing exploration and exploitation.  
* **Hyperband** improves search efficiency by using early stopping. It begins many training runs with a small budget (such as a few epochs), and progressively allocates more resources to the configurations that show early promise. Poor-performing trials are stopped early.  
* **Bayesian Optimization with Hyperband (BOHB)** integrates the probabilistic modeling of Bayesian optimization with the resource allocation mechanism of Hyperband. It uses a surrogate model to propose new configurations and evaluates them under the Hyperband scheduling scheme.

While all search strategies aim to optimize model performance by tuning hyperparameters, their effectiveness depends on the task and computational constraints. **Grid Search** is suitable for small, low-dimensional spaces, but becomes inefficient as dimensionality grows \[Bergstra & Bengio, 2012\]. **Random Search** offers better efficiency by sampling more diverse configurations under the same budget \[Bergstra & Bengio, 2012\]. **Bayesian Optimization** builds a surrogate model to guide the search, making it ideal for costly evaluations \[Snoek et al., 2012\]. **Hyperband** and **BOHB** enhance efficiency by combining early stopping with adaptive resource allocation \[Li et al., 2017; Falkner et al., 2018\].

In our case, we selected **Random Search** as a pragmatic strategy for conducting an initial exploratory sweep. Given the computational cost of each training run (approximately two hours due to the size and complexity of the astronomical image dataset), it was important to adopt a method that could explore the space effectively without requiring prior assumptions about parameter importance or the use of a surrogate model. The goal of this sweep is not to find the global optimum, but rather to gather early insights into the sensitivity and interaction of key hyperparameters within our specific detection task.

**Implementation**

Experiments were launched programmatically using wandb.agent, which executes multiple runs by sampling random configurations from the defined space. Each run is handled by a dynamically created wrapper function (sweep\_wrapper), responsible for initializing the W\&B session, extracting the current hyperparameter values, and invoking the training routine (train\_experiment) with those parameters.

To safeguard compute time and prevent inefficient runs, we implemented a custom speed guard mechanism. During initial testing, we observed that certain hyperparameter combinations—particularly very small learning rates or large batch sizes—led to prohibitively slow training, sometimes caused by instability or ineffective convergence. In response, we introduced a threshold-based mechanism that monitors the global iteration rate (in iterations per second).

This mechanism is implemented via the make\_speed\_guard function, which tracks the average training speed from the start of the run. If the iteration rate drops below a defined threshold (set to 0.8 it/s, based on empirical observations of typical training speeds), the run is automatically aborted. A message is logged to W\&B indicating the stop reason ("slow\_speed\_global"), and the process is cleanly terminated. This ensures that compute resources are not wasted on unproductive configurations.

The speed guard is injected into the training loop via the on\_batch\_end callback, and is evaluated once per batch. 

# 6\. Results {#6.-results}

## 6.1 Model comparison {#6.1-model-comparison}

Benchmarck  
→ Szimon with different encoders  
	Different backbones: how important it is to start with some pre-trained weights.   
	Table (model, metrics in validation test)

## Best performance {#best-performance}

Final choice Faster R-CNN v2 \+ ResNet-50

## 6.2 Effect of Non Maximum Suppression Threshold on Object Detection {#6.2-effect-of-non-maximum-suppression-threshold-on-object-detection}

To visualize the effect of non-maximum suppression in the telescope images, the test dataset was analyzed with 3 threshold values (0.3, 0.5 and 0.7) that are represented in the following images from top to bottom. 

In the following noisy image, it can be seen how by increasing the NMS threshold increases the number of predictions in the same object, clearly seen in the object cluster at (x,y) \~ (110,50), (250, 0), and (500, 0).

| ![][image31] |
| :---- |
| ![][image32] |
| ![][image33] |

The following example, less noisier, also shows how reducing the NMS threshold results in less redundant objects detected, such as in (x,y) \~ (220, 480\) or in (450,25).

| ![][image34] |
| :---- |
| ![][image35] |
| ![][image36] |

* 

## 6.3 Hyperparameter search

**Results**

WIP (no results until the sweep is done)

## 4\. Modifications implemented to the models / Special Techniques {#4.-modifications-implemented-to-the-models-/-special-techniques}

* 

* 

* 

#### **5\. Inference and Validation** {#5.-inference-and-validation}

* Final inferences performed on the cleaned dataset

* Validation set fixed (non-random) for consistency

* Results visualized and compared across architectures and settings

	  
→ Raul experiment with the Non Maximum Suppression threhsolds  
	Threshold of the maximum suppression with whatever we can \[0.3,0.5,0.7\]   
	Table showing validation and test scores \+ some images to tell the difference  
	Mask RCNN that was a different approach and did not include the image segmentation (optional)

# 

# CONCLUSIONS {#conclusions}

IMAGES SHOWING THE RESULTS  
	Same dataset with differents amount of data → try with differents croppings \[see what happens\]

Comment as an expert to tell about the quality metrics (not in the report) → questions, at the end

# Future improvements {#future-improvements}

- Zoom in dataset augmentation to make the model more sensible to size variable objects

## **Index** {#index}

1. **Introdution and motivation: Problem Statement**  
    1.1. Type of Images: Raw and Processed  
    1.2. Goal: Detecting Relevant Objects (Stars, Galaxies)  
    1.3. Challenge: Variability of Objects (Shiny, Wild, Saturated, Weird Cases)  
    1.4. Why It's Difficult: Noisy Backgrounds, Overexposed Areas, Non-trivial Morphologies  
    1.5. Reference to Classical Approaches (non-DL) and Their Limitation  
2. MILESTONES

3. **Methodology**  
    2.1. Data Preparation  
        2.1.1. Dataset Description  
        2.1.2. Ground Truth Creation (Manual Repetition, Annotation Decisions)  
        2.1.3. Dataset Size and Limitations  
        2.1.4. Train/Test/Validation Splits  
        2.1.5. Cropping Strategies and Motivation  
    2.2. Object Relevance Criteria  
        2.2.1. Excluded Cases: Saturated or Ambiguous Regions

4. **Model Experiments**  
    3.1. Baseline Models and Encoder Study (Szimon)  
        3.1.1. Backbone Architectures Used  
        3.1.2. Importance of Pre-trained Weights  
        3.1.3. Evaluation Table: Validation/Test Metrics  
    3.2. Suppression Threshold & Box Size Experiments (Raúl)  
        3.2.1. Non-Max Suppression Thresholds Tested (0.3, 0.5, 0.7)  
        3.2.2. Effect on Validation/Test Performance  
        3.2.3. Visual Examples to Illustrate Differences  
    3.3. Mask R-CNN Trial (Alternative Approach – Optional)

5. **Hyperparameter Tuning**  
    4.1. Current Model: Optimization Experiments (Cris & Gerard)  
    4.2. Parameters Explored  
        4.2.1. Batch Size  
        4.2.2. Learning Rate  
        4.2.3. Weight Decay  
        4.2.4. Early Stopping Patience (0, 3, 5 Epochs)  
    4.3. Performance Summary Table

6. **Conclusions**  
    5.1. Summary of Key Insights  
    5.2. Visual Results Across Models and Cropping Strategies  
    5.3. How the Dataset Size and Cropping Affected the Outcomes

7. **Expert Commentary (Optional, Not in Report)**  
    6.1. Reflections on Metric Quality (e.g. F1, IoU)  
    6.2. Questions for Future Work and Model Interpretation  
     
8. ANNEX\_  
   1. How to execute the code ( ) → Szimon 

EXAMPLE OF INDEX 

Introduction  
Motivation  
Milestones  
The data set  
Working Environment  
General Architecture →   
Main hyperparameters  
Metrics and loss criterions  
Preliminary Tests  
First steps  
Accessing the dataset  
Finding the right parameters  
Does the discriminator help?  
The quest for improving the results  
Increasing the pixel resolution of images  
Mid resolution  
High resolution  
Instance Normalization  
Data filtering  
VGG Loss  
Using the ReduceLROnPlateau scheduler  
Quality metrics  
Fréchet Inception Distance  
The Google Cloud instance  
Conclusions and Lessons Learned  
Next steps

# Bibliography  {#bibliography}

\[1\_cris\] Institut d'Estudis Espacials de Catalunya (IEEC). (n.d.). *Joan Oró Telescope*. Retrieved from [https://montsec.ieec.cat/en/joan-oro-telescope/](https://montsec.ieec.cat/en/joan-oro-telescope/)

\[2\_cris\] NASA Goddard Space Flight Center. (n.d.). *FITS Viewer*. Retrieved from [https://fits.gsfc.nasa.gov/fits\_viewer.html](https://fits.gsfc.nasa.gov/fits_viewer.html)

\[3\_cris\] Bertin, E., & Arnouts, S. (n.d.). *SExtractor Configuration Guide*. Retrieved from [https://sextractor.readthedocs.io/en/latest/Config.html](https://sextractor.readthedocs.io/en/latest/Config.html)

\[1\] Wu, C., Wong, O. I., Rudnick, L., Shabala, S. S., Alger, M. J., Banfield, J. K., Ong, C. S., White, S. V., Garon, A. F., Norris, R. P., Andernach, H., Tate, J., Lukic, V., Tang, H., Schawinski, K., & Diakogiannis, F. I. (2018). Radio Galaxy Zoo: CLARAN – A deep learning classifier for radio morphologies. Monthly Notices of the Royal Astronomical Society, 482(1), 1211–1230. https://doi.org/10.1093/mnras/sty2646  
\[2\]  Burke, C. J., Aleo, P. D., Chen, Y.-C., Liu, X., Peterson, J. R., Sembroski, G. H., & Lin, J. Y.-Y. (2019). Deblending and classifying astronomical sources with Mask R-CNN deep learning. Monthly Notices of the Royal Astronomical Society, 490(3), 3952–3965. [https://doi.org/10.1093/mnras/stz2845](https://doi.org/10.1093/mnras/stz2845)

\[3\] Sortino, R., Magro, D., Fiameni, G., Sciacca, E., Riggi, S., DeMarco, A., Spampinato, C., Hopkins, A. M., Bufano, F., Schillirò, F., Bordiu, C., & Pino, C. (2023). Radio astronomical images object detection and segmentation: A benchmark on deep learning methods. Experimental Astronomy, 56(1), 293–331. [https://doi.org/10.1007/s10686-023-09893-w](https://doi.org/10.1007/s10686-023-09893-w)

\[4\] H. Farias, D. Ortiz, G. Damke, M. Jaque Arancibia, M. Solar, Mask galaxy: Morphological segmentation of galaxies, Astronomy and Computing, Volume 33, 2020, 100420, ISSN 2213-1337, [https://doi.org/10.1016/j.ascom.2020.100420](https://doi.org/10.1016/j.ascom.2020.100420).

\[5\] Bergstra, J., & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization*. Journal of Machine Learning Research, 13, 281–305. https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf

\[6\] Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian Optimization of Machine Learning Algorithms*. In *Advances in Neural Information Processing Systems* (pp. 2951–2959). [https://arxiv.org/abs/1206.2944](https://arxiv.org/abs/1206.2944)

\[7\] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*. Journal of Machine Learning Research, 18(185), 1–52. [https://arxiv.org/abs/1603.06560](https://arxiv.org/abs/1603.06560)

\[8\] Falkner, S., Klein, A., & Hutter, F. (2018). *BOHB: Robust and Efficient Hyperparameter Optimization at Scale*. In Proceedings of the 35th International Conference on Machine Learning (ICML 2018). [https://arxiv.org/abs/1807.01774](https://arxiv.org/abs/1807.01774)

\[Farias, 2020\] H. Farias, D. Ortiz, G. Damke, M. Jaque Arancibia, and M. Solar, “Mask galaxy: Morphological segmentation of galaxies,” Astronomy and Computing, vol. 33, p. 100420, Oct. 2020, doi: 10.1016/j.ascom.2020.100420.

\[He, 2023\] Y. He, J. Wu, W. Wang, B. Jiang, and Y. Zhang, “AstroYOLO: A hybrid CNN–Transformer deep-learning object-detection model for blue horizontal-branch stars,” Publications of the Astronomical Society of Japan, vol. 75, no. 6, pp. 1311–1323, Oct. 2023, doi: 10.1093/pasj/psad071.

\[Long, 2024\] M. Long, X. Jiarong, D. Jiangbin, Z. Jiayao, W. Xiaotian, and Z. Yu, “Astronomical Pointlike Source Detection via Deep Feature Matching,” ApJS, vol. 276, no. 1, p. 4, Dec. 2024, doi: 10.3847/1538-4365/ad9244.

\[Min, 2022\] K. Min, G.-H. Lee, and S.-W. Lee, “Attentional feature pyramid network for small object detection,” Neural Networks, vol. 155, pp. 439–450, Nov. 2022, doi: 10.1016/j.neunet.2022.08.029.

\[Wang, 2023\] X. Wang, G. Wei, S. Chen, and J. Liu, “An efficient weakly semi-supervised method for object automated annotation,” Multimed Tools Appl, vol. 83, no. 3, pp. 9417–9440, June 2023, doi: 10.1007/s11042-023-15305-0.

\[Wu, 2020\] T. Wu, “A Supernova Detection Implementation based on Faster R-CNN,” 2020 International Conference on Big Data \&amp; Artificial Intelligence \&amp; Software Engineering (ICBASE). IEEE, pp. 390–393, Oct. 2020\. doi: 10.1109/icbase51474.2020.00089.

\[Xiao, 2025\] Y. Xiao, Y. Guo, Q. Pang, X. Yang, Z. Zhao, and X. Yin, “STar-DETR: A Lightweight Real-Time Detection Transformer for Space Targets in Optical Sensor Systems,” Sensors, vol. 25, no. 4, p. 1146, Feb. 2025, doi: 10.3390/s25041146.

\[Zhang, 2025\] Zhang, D., Feng, T., Xue, L., Wang, Y., Dong, Y., & Tang, J. (2025). Parameter-efficient fine-tuning for foundation models. arXiv preprint arXiv:2501.13787.  
