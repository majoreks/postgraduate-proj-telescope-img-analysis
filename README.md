# postgraduate-proj-telescope-img-analysis

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
Additionally to requirements specified above, user should be logged into wandb locally for logging purposes (if not dev run).

```bash
python main.py --mode train --task <task-name> [--dev]
```
- `--task` parameter is required and defines the task name for logging purposes
- `--dev` flag can be additionally passed, this will ensure that the wandb logging is not enabled, only subset of images is used and few epochs are run.   

### Inference
```bash
python main.py --mode infer --task <task-name>
```
In order to run inference instead of training, `--mode` should be set to `infer`, `--task` flag is still required, however has no real effect so can have any value. Model weights are automatically downloaded by the inference script, path to test images is found in data_path configured in config file and then looking inside `test_dataset` directory inside of directory indicated by data_path

## Report
[link to report in the repo]