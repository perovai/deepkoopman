import matplotlib.pyplot as plt
import numpy as np
import torch

from koop.model import DeepKoopman
from koop.utils import resolve


def plot_2D_comparative_trajectories(
    model: DeepKoopman,
    batch,
    n_steps,
    traj_per_plot=5,
    exp=None,
    fig_path="./2D_traj.png",
    show=False,
):

    fig_path = resolve(fig_path)
    if traj_per_plot <= 0:
        traj_per_plot = len(batch)

    with torch.no_grad():
        initial_states = batch[:, 0, ...]
        _, intermediate_states = model.infer_trajectory(initial_states, n_steps)

    trajectories = initial_states.clone().unsqueeze(1)
    intermediate_states = torch.cat([s.unsqueeze(1) for s in intermediate_states], 1)
    # batch x time x dim
    trajectories = torch.cat([trajectories, intermediate_states], 1).cpu().numpy()
    batch = batch.cpu().numpy()
    colors = plt.cm.jet(np.linspace(0, 1, len(initial_states)))

    k = 0
    plots = np.ceil(len(batch) / traj_per_plot)

    for p in range(int(plots)):
        plt.figure()
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), dpi=300, sharex=True, sharey=True)
        for _ in range(traj_per_plot):

            axs[0].scatter(
                trajectories[k, :, 0],
                trajectories[k, :, 1],
                color=colors[k],
                marker="o",
                alpha=0.5,
            )
            axs[0].set_title("Ground Truth")
            axs[0].axhline(y=0)
            axs[0].axvline(x=0)

            axs[1].scatter(
                batch[k, : (n_steps + 1), 0],
                batch[k, : (n_steps + 1), 1],
                color=colors[k],
                marker="x",
                alpha=0.5,
            )
            axs[1].set_title("Predictions")
            axs[1].axhline(y=0)
            axs[1].axvline(x=0)

            k += 1
            if k >= len(batch):
                break

        fp = fig_path.parent / (fig_path.stem + f"_{p}" + fig_path.suffix)
        fig.savefig(fp)
        if exp is not None:
            exp.log_image(fp)

    if exp is None and show:
        plt.show()
