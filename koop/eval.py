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
        for _ in range(traj_per_plot):

            plt.scatter(
                trajectories[k, :, 0],
                trajectories[k, :, 1],
                color=colors[k],
                marker="o",
                alpha=0.3,
            )
            plt.scatter(
                batch[k, : (n_steps + 1), 0],
                batch[k, : (n_steps + 1), 1],
                color=colors[k],
                marker="x",
                alpha=0.3,
            )
            # breakpoint()
            k += 1
            if k >= len(batch):
                break
        plt.title("x -> Ground Truth | • -> Predictions")
        fp = fig_path.parent / (fig_path.stem + f"_{p}" + fig_path.suffix)
        plt.savefig(fp, dpi=300)
        if exp is not None:
            exp.log_image(fp)
    if exp is None and show:
        plt.show()
