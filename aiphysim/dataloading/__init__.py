from pathlib import Path

from torch.utils.data import DataLoader

from aiphysim.utils import load_opts, resolve

from .dataloader_spacetime import RB2DataLoader
from .density_dataset import DatDensityDataset, H5DensityDataset
from .koopman_dataset import KoopmanDataset


def create_datasets(opts):

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

    if dataset_type == "datdensity":
        train_files = list(Path(path).glob("train_*.json"))
        val_files = list(Path(path).glob("val_*.json"))

        return {
            "train": DatDensityDataset(train_files, lims["train"], force_rebase),
            "val": DatDensityDataset(val_files, lims["val"], force_rebase),
        }

    if dataset_type == "spacetime":
        dataset_file = list(Path(path).glob("snapshots.h5"))[0]

        return {
            "train": RB2DataLoader(path, dataset_file, lims["train"]),
            "val": RB2DataLoader(path, dataset_file, lims["val"]),
            "test": RB2DataLoader(path, dataset_file, lims["test"]),
        }

    raise ValueError("Unknown dataset type: " + str(dataset_type))


def create_dataloaders(opts, verbose=1):
    """
    Return a dict of data loaders with keys train/val/test

    Args:
        path (pathlike): Path to read the data from
        sequence_length (int): How to reshape the data from N x dim to B x seq x dim
        batch_size (int): Batch size

    Returns:
        dict: {"train": dataloader, "val": dataloader, "test": dataloader}
    """

    datasets = create_datasets(opts)

    workers = opts.workers
    batch_size = opts.batch_size

    loaders = {
        k: DataLoader(
            v, batch_size, shuffle=k == "train", pin_memory=True, num_workers=workers
        )
        for k, v in datasets.items()
    }

    if verbose > 0:
        for mode, loader in loaders.items():
            print()
            print(f"Found {len(loader.dataset)} samples in the {mode} dataset")
        print()

    return loaders
