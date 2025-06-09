from collections import defaultdict
from datetime import datetime
import wandb
import torch.nn as nn
from postprocess.plot_losses import plot_losses

PROJECT_NAME = "postgraduate-sat-object-detection"

class Logger():
    def __init__(self, task: str, config: dict, dev: bool = False):
        self._enabled = not dev

        if self._enabled:
            self.__init_logger(task, config)
        else:
            self.__eval_loss = defaultdict(list)
            self.__train_loss = defaultdict(list)

    def __init_logger(self, task: str, config: dict) -> None:
        wandb.login()
        wandb.init(project=PROJECT_NAME, config=config)
        wandb.run.name = f'{task}-{datetime.now().strftime("%d/%m/%Y-%H:%M")}'

    def log_model(self, model: nn.Module) -> None:
        if self._enabled:
            wandb.watch(model, log='all', log_freq=100, log_graph=True)

    def log_train_loss(self, loss_dict: dict, is_train: bool) -> None:
        if self._enabled:
            label = 'training' if is_train else 'eval'
            log_data = {f'{label}/{k}': v.item() for k, v in loss_dict.items()}
            wandb.log(log_data)
        else:
            aggregated_loss_dict = self.__train_loss if is_train else self.__eval_loss
            for k, v in loss_dict.items():
                aggregated_loss_dict[k].append(v.item())

    def flush(self) -> None:
        if not self._enabled:
            plot_losses(self.__train_loss, fname="train_loss.png", save_plot=True)
            plot_losses(self.__eval_loss, fname="eval_loss.png", save_plot=True)