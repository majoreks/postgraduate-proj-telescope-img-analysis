# postgraduate-proj-telescope-img-analysis

## Report
[Report link](https://github.com/majoreks/postgraduate-proj-telescope-img-analysis/blob/main/report/report.md) (report/report.md)

## How to run
Pre requirement to run the training or inference is installing necessary requirements. This can be done by running (assuming `pip` usage)
```bash
pip install -r requirements.txt
```   
Once requirements are installed, the main part of the project can be executed via the `main.py` script. Before exeuction `config` in `main.py` should be adapted by 
- data of interest should be unpacked in directory specified by `data_path` 
- output directory specified by `output_path` should be created.   

Project can be run in train mode and inference mode.

### Training

Before running, make sure:

- Your environment meets the requirements listed above.
- You’re logged into Weights & Biases locally (unless using `--dev`).

#### Required flags

- `--mode` / `-m`  
  Must be set to `train`.

- `--task` / `-t`  
  A unique name for this training run (used for W&B logging).

#### Optional flags

- `--dev`  
  If present, enables development mode:
  - Disables W&B logging
  - Uses a smaller subset of data
  - Runs for fewer epochs

- `--resnet-type`  
  Choose a ResNet backbone to initialize only the backbone weights (ImageNet pretrained). One of:
  - `resnet18`
  - `resnet34`
  - `resnet50`
  - `resnet101`  
  If omitted, the script will assume ResNet50 as backbone and weights for the entire network will be loaded.

- `--model-type`  
  Which Faster R-CNN variant to use. One of:
  - `v1`
  - `v2`  
  Defaults to `v2` if not provided.

#### Basic example

Running with particular resnet as backbone  

```bash
python main.py \
  --mode train \
  --task my_experiment \
  --resnet-type resnet34 \
  --model-type v1
``` 

Running in default, where FasterRCNN v2 with ResNet50 backbone will be assumed and pretrained weights for the whole network will be used

```bash
python main.py \
  --mode train \
  --task my_experiment
``` 

### Inference

To run the model in inference mode, you need to supply:

- `--mode` / `-m`  
  Must be set to `infer`.

- `--task` / `-t`  
  Output directory where your inference outputs (e.g., visualizations, JSON results) will be saved.

- `--weights-path`  
  **(required)** Path to the `.pth` or `.pt` file containing your pre-trained Faster R-CNN model weights.

- `--resnet-type`  
  **(required)** Which ResNet backbone your model uses. One of:
  - `resnet18`
  - `resnet34`
  - `resnet50`
  - `resnet101`

- `--model-type`  
  **(required)** Which Faster R-CNN variant to use. One of:
  - `v1`
  - `v2`

The script will automatically discover your test images by looking under the `test_dataset` folder in the `data_path` configured in your `config` file.

#### Basic example

```bash
python main.py \
  --mode infer \
  --task inference_run_01 \
  --weights-path /path/to/model_resnet50_v2.pth \
  --resnet-type resnet50 \
  --model-type v2
```
