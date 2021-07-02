"""RB2 Experiment Dataloader"""
import os

import h5py
import numpy as np
from torch.utils.data import Dataset

# pylint: disable=too-manz-arguments, too-manz-instance-attributes, too-manz-locals


class RB2DataLoader(Dataset):
    """Pytorch Dataset instance for loading Rayleigh Bernard 2D dataset.

    Loads a 2d space + time cubic cutout from the whole simulation.
    """

    def __init__(
        self,
        data_dir="./",
        data_filename="./data/snapshots.h5",
        mode="train",
        res_x=64,
        res_z=64,
        train_size=0.7,
        val_size=0.2,
        normalize=True,
        timesteps=4,
    ):
        """

        Initialize DataSet
        Args:
          data_dir: str, path to the dataset folder, default="./"
          data_filename: str, name of the dataset file, default="/data/snapshots.h5"
          mode: str, dataset mode, can be on of ['train', 'val', 'test'], default="train"
          res_x: int, resolution in the x-axis, default=64
          res_z: int, resolution in the z-axis, default=64
          train_size: float, specifies the portion of the dataset to be held for training, default=0.7
          val_size: float, specifies the portion of the dataset to be held for validation, default=0.2
          normalize: bool, whether to standardize the data or not, default=True
          timesteps: int, number of past time-steps to consider for the input, default=4
        """

        assert mode in [
            "train",
            "val",
            "test",
        ], "mode must belong to one of these ['train', 'val', 'test']"
        assert train_size > 0 and train_size < 1, "train_size should be between 0 and 1"
        assert val_size > 0 and val_size < 1, "val_size should be between 0 and 1"

        self.data_dir = data_dir
        self.data_filename = data_filename
        self.mode = mode
        self.res_x = res_x
        self.res_z = res_z
        self.normalize = normalize
        self.timesteps = timesteps

        # concatenating pressure, temperature, x-velocity, and z-velocity as a 4 channel array: pbuw
        hdata = h5py.File(os.path.join(self.data_dir, self.data_filename), "r")

        nt, _, _ = hdata["tasks"]["u"].shape

        if mode == "train":
            self.data = np.stack(
                [
                    hdata["tasks"]["p"][: int(nt * train_size)],
                    hdata["tasks"]["T"][: int(nt * train_size)],
                    hdata["tasks"]["u"][: int(nt * train_size)],
                    hdata["tasks"]["w"][: int(nt * train_size)],
                ],
                axis=0,
            )
        elif mode == "val":
            self.data = np.stack(
                [
                    hdata["tasks"]["p"][
                        int(nt * train_size) : int(nt * (train_size + val_size))
                    ],
                    hdata["tasks"]["T"][
                        int(nt * train_size) : int(nt * (train_size + val_size))
                    ],
                    hdata["tasks"]["u"][
                        int(nt * train_size) : int(nt * (train_size + val_size))
                    ],
                    hdata["tasks"]["w"][
                        int(nt * train_size) : int(nt * (train_size + val_size))
                    ],
                ],
                axis=0,
            )
        else:
            self.data = np.stack(
                [
                    hdata["tasks"]["p"][int(nt * (train_size + val_size)) :],
                    hdata["tasks"]["T"][int(nt * (train_size + val_size)) :],
                    hdata["tasks"]["u"][int(nt * (train_size + val_size)) :],
                    hdata["tasks"]["w"][int(nt * (train_size + val_size)) :],
                ],
                axis=0,
            )

        self.data = self.data.astype(np.float32)
        self.data = self.data.transpose(1, 0, 3, 2)  # [t, c, z, x]
        self.nt_data, self.nc_data, self.nz_data, self.nx_data = self.data.shape

        self.nx_start_range = np.arange(0, self.nx_data - self.res_x + 1)
        self.nz_start_range = np.arange(0, self.nz_data - self.res_z + 1)
        self.nt_start_range = np.arange(0, self.nt_data - self.timesteps + 1)
        self.rand_grid = np.stack(
            np.meshgrid(
                self.nt_start_range,
                self.nz_start_range,
                self.nx_start_range,
                indexing="ij",
            ),
            axis=-1,
        )
        # (xaug, zaug, taug, 3)
        self.rand_start_id = self.rand_grid.reshape([-1, 3])

        # compute channel-wise mean and std
        self._mean = np.mean(self.data, axis=(0, 2, 3))
        self._std = np.std(self.data, axis=(0, 2, 3))

    def __len__(self):
        return self.rand_start_id.shape[0] - self.nt_data

    def __getitem__(self, idx):
        t_id, z_id, x_id = self.rand_start_id[idx]
        u_t = self.data[
            t_id : t_id + self.timesteps,
            :,
            z_id : z_id + self.res_z,
            x_id : x_id + self.res_x,
        ]  # [t, c, z, x]
        u_t_next = self.data[
            t_id + self.timesteps : t_id + self.timesteps + 1,
            :,
            z_id : z_id + self.res_z,
            x_id : x_id + self.res_x,
        ]  # [t, c, z, x]

        return_tensors = [u_t, u_t_next]
        # cast everything to float32
        if self.normalize:
            transposed_mean = self._mean[(None,) + (...,) + (None,) * 2]
            transposed_std = self._std[(None,) + (...,) + (None,) * 2]
            return_tensors = [
                ((t - transposed_mean) / transposed_std).astype(np.float32)
                for t in return_tensors
            ]
        else:
            return_tensors = [t.astype(np.float32) for t in return_tensors]
        return tuple(return_tensors)
