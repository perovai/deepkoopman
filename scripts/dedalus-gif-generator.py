import argparse
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

import matplotlib.animation as animation


def get_args():
    parser = argparse.ArgumentParser(
        description="Data generation script for 2D Rayleigh-Benard convection using Dedalus"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to Dataset file in h5 format to be animated",
    )
    parser.add_argument(
        "--variable",
        default='T',
        type=str,
        help="Specify which variable you want displayed, one of [T,p,u,w]. (default: 'T')",
    )
    parser.add_argument(
        "--dst",
        default='.',
        type=str,
        help="Path to Destination directory. (default: '.')",
    )
    parser.add_argument(
        "--fps",
        default=24,
        type=int,
        help="Number of frames per second. (default: 24)",
    )
    parser.add_argument(
        "--cmap",
        default='RdBu_r',
        type=str,
        help="Colormap of the displayed quantities. (default: 'RdBu_r')",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    fps = args.fps
    path = args.file
    cmap = args.cmap
    dst = args.dst
    phy_var = args.variable

    assert phy_var in ['T', 'u', 'p', 'w'], "Physical variable should be one of ['T', 'p', 'u', 'w']"
    data = h5py.File(path, 'r')
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(8, 4))

    variable = np.transpose(data['tasks'][phy_var][:], (0, 2, 1))
    n_frames = variable.shape[0]
    im = plt.imshow(variable[0], cmap=cmap)
    plt.colorbar(im)

    def animate_func(i):
        im.set_array(variable[i])
        return [im]
    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = n_frames,
                                interval = 1000 / fps, # in ms
                                )
    anim.save(os.path.join(args.dst, f'animation_{phy_var}.gif'), fps=fps)