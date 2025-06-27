import matplotlib.pyplot as plt
import math

def plot_losses(loss_history, fname, save_plot = False, is_loss = True) -> None:
    num_losses = len(loss_history)
    cols = 2  # or 3 if you prefer more compact layout
    rows = math.ceil(num_losses / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows), constrained_layout=True)
    axs = axs.flatten()

    for i, (loss_name, values) in enumerate(loss_history.items()):
        axs[i].plot(values, label=loss_name, color='tab:blue')
        axs[i].set_title(loss_name, fontsize=12)
        axs[i].set_xlabel("Batch Steps", fontsize=10)
        axs[i].set_ylabel("Loss" if is_loss else "Metric", fontsize=10)
        axs[i].legend()
        axs[i].grid(True, linestyle='--', alpha=0.5)

    # Turn off any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    fig.suptitle("Training loss components over time" if is_loss else "Eval metric components over time", fontsize=16)

    if save_plot:
        plt.savefig(f'output/{fname}', dpi=400)
    else:
        plt.show()