from collections import defaultdict
from datetime import datetime
import torch
import wandb
import torch.nn as nn
from postprocess.plot_losses import plot_losses

PROJECT_NAME = "postgraduate-sat-object-detection"

class Logger():
    def __init__(self, task: str, config: dict, dev: bool = False):
        self._enabled = not dev
        self._step = 1

        if self._enabled:
            self.__init_logger(task, config)
        else:
            self.__eval_loss = defaultdict(list)
            self.__train_loss = defaultdict(list)

    def __init_logger(self, task: str, config: dict) -> None:
        # For the experiment mode to run
        if wandb.run is not None and wandb.run._settings._run_id is not None:
            return
        
        # For the train mode to run
        wandb.login()
        wandb.init(project=PROJECT_NAME, config=config)
        wandb.run.name = f'{task}-{datetime.now().strftime("%d/%m/%Y-%H:%M")}'

    def log_model(self, model: nn.Module) -> None:
        if self._enabled:
            wandb.watch(model, log='all', log_freq=100, log_graph=True)

    def log_train_loss(self, *loss_dicts: dict, is_train: bool) -> None:
        loss_dict = { k: v for d in loss_dicts for k, v in d.items() }

        if self._enabled:
            label = 'training' if is_train else 'eval'
            log_data = {}

            for k, v in loss_dict.items():
                # 🔧 Solución robusta: convierte todo a tensor si no lo es
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)

                if v.numel() == 1:
                    log_data[f'{label}/{k}'] = v.item()
                else:
                    for i, val in enumerate(v.tolist()):
                        log_data[f'{label}/{k}_{i}'] = val

            wandb.log(log_data, step=self._step)

        else:
            aggregated_loss_dict = self.__train_loss if is_train else self.__eval_loss

            for k, v in loss_dict.items():
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)

                if v.numel() == 1:
                    aggregated_loss_dict[k].append(v.item())
                else:
                    for i, val in enumerate(v.tolist()):
                        aggregated_key = f"{k}_{i}"
                        if aggregated_key not in aggregated_loss_dict:
                            aggregated_loss_dict[aggregated_key] = []
                        aggregated_loss_dict[aggregated_key].append(val)

    def step(self) -> None:
        self._step += 1

    def flush(self) -> None:
        if not self._enabled:
            plot_losses(self.__train_loss, fname="train_loss.png", save_plot=True)
            plot_losses(self.__eval_loss, fname="eval_loss.png", save_plot=True, is_loss=False)

    def log_text(self, msg: str) -> None:
        if self._enabled:
            wandb.log({"log/text": msg}, step=self._step)
        else:
            print(msg)

    def log_early_stop(self) -> None:
        wandb.summary.update({ "early_stop": True })

    def config_status(self) -> dict:
        return wandb.config