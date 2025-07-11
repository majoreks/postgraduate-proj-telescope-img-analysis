import wandb
from train.traineval import train_experiment

def sweep_wrapper_factory(base_config, sweep_config, task, dev, device, tempdir):
    def sweep_wrapper():
        sweep_params = dict(wandb.config)

        # Mezclar base_config con sweep_params
        merged_config = base_config.copy()
        merged_config.update(sweep_params)


        train_experiment(merged_config, tempdir, task, dev, device)
    
    return sweep_wrapper