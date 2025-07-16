import wandb, time, sys
from train.traineval import train_experiment

# ──────────────────────────────────────────────────────────────
def make_speed_guard(threshold_it_s: float):
    """
    Corta el run si la velocidad media global (igual a tqdm) baja del umbral.
    """
    t_start      = None   
    global_steps = 0

    def _guard():
        nonlocal t_start, global_steps
        global_steps += 1
        if t_start is None:          
            t_start = time.time()
            return                    

        elapsed = time.time() - t_start
        it_per_s = global_steps / elapsed     
        wandb.log({"it_per_s_global": it_per_s}, commit=False)

        if it_per_s < threshold_it_s:
            wandb.log({"stop_reason": "slow_speed_global", "it_per_s": it_per_s})
            print(f"🛑 Velocidad global {it_per_s:.2f} it/s < {threshold_it_s} → aborting run")
            wandb.finish()          
            sys.exit(0)

    return _guard



def sweep_wrapper_factory(base_cfg, sweep_cfg, task, dev, device, tempdir):
    def sweep_wrapper():
        run = wandb.init(
            project="postgraduate-sat-object-detection",
            config=sweep_cfg,
            tags=["hyperparam-sweep"]
        )
        sweep_params = dict(wandb.config)

        speed_guard = make_speed_guard(threshold_it_s=0.8)  
        train_experiment(
            base_cfg, tempdir, task, dev, device,
            sweep_params,
            on_batch_end=speed_guard
        )

        wandb.finish()
    return sweep_wrapper