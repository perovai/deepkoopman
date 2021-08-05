from pathlib import Path
import random

import h5py
import numpy as np
from torch.utils.data import DataLoader

from aiphysim.utils import resolve, temp_seed

from .dataloader_spacetime import RB2DataLoader
from .density_dataset import DatDensityDataset, H5DensityDataset, SplitH5DensityDataset
from .koopman_dataset import KoopmanDataset


def create_datasets(opts):
    """
    Creates the appropriate dataset according to opts.dataset_type

    Args:
        opts (addict.Dict): options dictionnary

    Raises:
        ValueError: If the dataset_type is unknown

    Returns:
        dict: dictionnary mapping mode (train/val/test) to dataset
    """

    lims = {
        f"{mode}": opts.get("limit", {}).get(mode, -1)
        for mode in ["train", "val", "test"]
    }

    path = resolve(opts.data_folder)
    sequence_length = opts.sequence_length
    dataset_type = opts.dataset_type
    force_rebase = opts.get("force_rebase")

    if dataset_type == "koopman":
        print("Creating datasets from ", str(path))
        train_files = list(Path(path).glob("*_train*.csv"))
        val_files = list(Path(path).glob("*_val*.csv"))
        test_files = list(Path(path).glob("*_test*.csv"))

        return {
            "train": KoopmanDataset(train_files, sequence_length, lims["train"]),
            "val": KoopmanDataset(val_files, sequence_length, lims["val"]),
            "test": KoopmanDataset(test_files, sequence_length, lims["test"]),
        }

    if dataset_type == "h5density":
        train_files = list(Path(path).glob("train_*.h5"))
        val_files = list(Path(path).glob("val_*.h5"))

        return {
            "train": H5DensityDataset(train_files, lims["train"]),
            "val": H5DensityDataset(val_files, lims["val"]),
        }

    if dataset_type == "splith5density":
        n_samples = -1
        h5_path = resolve(opts.data_file)
        with h5py.File(h5_path, "r") as archive:
            n_samples = len(archive)

        with temp_seed(123):
            indices = np.random.permutation(n_samples)

        train_indices = indices[: int(opts.train_ratio * n_samples)]
        val_indices = indices[int(opts.train_ratio * n_samples) :]

        return {
            "train": SplitH5DensityDataset(h5_path, train_indices, lims["train"]),
            "val": SplitH5DensityDataset(h5_path, val_indices, lims["val"]),
        }

    if dataset_type == "datdensity":
        train_files = list(Path(path).glob("train_*.json"))
        val_files = list(Path(path).glob("val_*.json"))

        return {
            "train": DatDensityDataset(train_files, lims["train"], force_rebase),
            "val": DatDensityDataset(val_files, lims["val"], force_rebase),
        }

    if dataset_type == "spacetime":
        if "dataset_file" in opts:
            dataset_file = opts.dataset_file
        else:
            dataset_file = "snapshots.h5"
        ratios = {
            f"{mode}": opts.get("ratio", {}).get(mode, -1) for mode in ["train", "val"]
        }

        if "normalize" in opts:
            normalize = opts.normalize
        else:
            normalize = True

        if "timesteps" in opts:
            timesteps = opts.timesteps
        else:
            raise Exception("You should provide a value of 'timesteps' in the yaml configuration file!")

        return {
            "train": RB2DataLoader(
                path,
                dataset_file,
                "train",
                ratios["train"],
                ratios["val"],
                normalize,
                timesteps,
            ),
            "val": RB2DataLoader(
                path,
                dataset_file,
                "val",
                ratios["train"],
                ratios["val"],
                normalize,
                timesteps,
            ),
            "test": RB2DataLoader(
                path,
                dataset_file,
                "test",
                ratios["train"],
                ratios["val"],
                normalize,
                timesteps,
            ),
        }

    raise ValueError("Unknown dataset type: " + str(dataset_type))


def create_dataloaders(opts, verbose=1):
    """
    Return a dict of data loaders with keys train/val/test

    Args:
        opts (addict.Dict): options dictionnary
        verbose (int): whether or not to print dataset lengths

    Returns:
        dict: dictionnary mapping mode (train/val/test) to dataloader
    """
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    datasets = create_datasets(opts)

    workers = opts.workers
    batch_size = opts.batch_size

    loaders = {
        k: DataLoader(
            v, batch_size, shuffle=k == "train", pin_memory=True, num_workers=workers, worker_init_fn=seed_worker
        )
        for k, v in datasets.items()
    }

    if verbose > 0:
        print()
        for mode, loader in loaders.items():
            print(f"Found {len(loader.dataset)} samples in the {mode} dataset")
        print()

    return loaders
