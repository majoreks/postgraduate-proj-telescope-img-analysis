import wandb
from train.traineval import train_experiment
from logger.logger import Logger

def sweep_wrapper_factory(base_config, sweep_config, task, dev, device, tempdir):
    def sweep_wrapper():   
        run = wandb.init(project="postgraduate-sat-object-detection", config=sweep_config)
        sweep_params = dict(wandb.config)                 
        train_experiment(base_config, tempdir, task, dev, device, sweep_params)
    return sweep_wrapper