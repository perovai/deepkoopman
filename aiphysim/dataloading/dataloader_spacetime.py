"""RB2 Experiment Dataloader"""
import os

import h5py
import numpy as np
from torch.utils.data import Dataset

# pylint: disable=too-many-arguments, too-many-instance-attributes, too-many-locals


class RB2DataLoader(Dataset):
    """Pytorch Dataset instance for loading Rayleigh Bernard 2D dataset.

    Loads a 2d space + time cubic cutout from the whole simulation.
    """

    def __init__(
        self,
        data_dir="./",
        data_filename="./data/snapshots.h5",
        mode="train",
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
        self.normalize = normalize
        self.timesteps = timesteps

        hdata = h5py.File(os.path.join(self.data_dir, self.data_filename), "r")

        nt = hdata["tasks"]["u"].shape[0]

        mean, std = self.compute_statistics(hdata, train_size)

        # concatenating pressure, temperature, x-velocity, and z-velocity as a 4 channel array: pbuw
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
            self.data = self.data.transpose(1, 0, 3, 2)  # [t, c, z, x]

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

            self.data = self.data.transpose(1, 0, 3, 2)  # [t, c, z, x]
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

            self.data = self.data.transpose(1, 0, 3, 2)  # [t, c, z, x]

        self.nt_data = self.data.shape[0]

        self._mean = mean
        self._std = std

    def compute_statistics(self, hdata, train_size):
        """
        Computes the mean and standard deviation from the training set's data

        Args:
            hdata : h5py.Dataset, The h5 dataset
            train_size : float, specifies the portion of the dataset to be held for training
        """
        nt = hdata["tasks"]["u"].shape[0]

        training_data = np.stack(
            [
                hdata["tasks"]["p"][: int(nt * train_size)],
                hdata["tasks"]["T"][: int(nt * train_size)],
                hdata["tasks"]["u"][: int(nt * train_size)],
                hdata["tasks"]["w"][: int(nt * train_size)],
            ],
            axis=0,
        )
        training_data = training_data.transpose(1, 0, 3, 2)
        mean = np.mean(training_data, axis=(0, 2, 3))
        std = np.std(training_data, axis=(0, 2, 3))
        return mean, std

    def __len__(self):
        return self.nt_data - self.timesteps

    def __getitem__(self, idx):
        u_t = self.data[idx : idx + self.timesteps]  # [t, c, z, x]
        u_t_next = self.data[
            idx + self.timesteps : idx + self.timesteps + 1
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
