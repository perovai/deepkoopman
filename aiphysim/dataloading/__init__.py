from pathlib import Path

from torch.utils.data import DataLoader

from .density_dataset import DatDensityDataset, H5DensityDataset
from .dataloader_spacetime import RB2DataLoader
from .koopman_dataset import KoopmanDataset


def create_datasets(
    path,
    sequence_length,
    dataset_type="koopman",
    force_rebase=None,
    train_lim=-1,
    val_lim=-1,
    test_lim=-1,
):
    if dataset_type == "koopman":
        print("Creating datasets from ", str(path))
        train_files = list(Path(path).glob("*_train*.csv"))
        val_files = list(Path(path).glob("*_val*.csv"))
        test_files = list(Path(path).glob("*_test*.csv"))

        return {
            "train": KoopmanDataset(train_files, sequence_length, train_lim),
            "val": KoopmanDataset(val_files, sequence_length, val_lim),
            "test": KoopmanDataset(test_files, sequence_length, test_lim),
        }

    if dataset_type == "h5density":
        train_files = list(Path(path).glob("train_*.h5"))
        val_files = list(Path(path).glob("val_*.h5"))

        return {
            "train": H5DensityDataset(train_files, train_lim),
            "val": H5DensityDataset(val_files, val_lim),
        }

    if dataset_type == "datdensity":
        train_files = list(Path(path).glob("train_*.json"))
        val_files = list(Path(path).glob("val_*.json"))

        return {
            "train": DatDensityDataset(train_files, train_lim, force_rebase),
            "val": DatDensityDataset(val_files, val_lim, force_rebase),
        }

    if dataset_type == "spacetime":
        # Sample dataset for setting up code
        train_files = list(Path(path).glob("snapshots_s1_p0.h5"))

        return {
            "train": RB2DataLoader(
                train_files, train_lim
            ),  # Class from implementation of Meshfree Flow Net paper
        }

    raise ValueError("Unknown dataset type: " + str(dataset_type))


def create_dataloaders(opts):
    """
    Return a dict of data loaders with keys train/val/test

    Args:
        path (pathlike): Path to read the data from
        sequence_length (int): How to reshape the data from N x dim to B x seq x dim
        batch_size (int): Batch size

    Returns:
        dict: {"train": dataloader, "val": dataloader, "test": dataloader}
    """

    lims = {
        f"{mode}": opts.get("limit", {}).get(mode, -1)
        for mode in ["train", "val", "test"]
    }

    path = opts.data_folder
    sequence_length = opts.sequence_length
    batch_size = opts.batch_size
    dataset_type = opts.dataset_type
    workers = opts.workers
    force_rebase = opts.get("force_rebase")

    path = Path(path).expanduser().resolve()
    datasets = create_datasets(
        path,
        sequence_length,
        dataset_type,
        force_rebase,
        lims["train"],
        lims["val"],
        lims["test"],
    )
    return {
        k: DataLoader(
            v, batch_size, shuffle=k == "train", pin_memory=True, num_workers=workers
        )
        for k, v in datasets.items()
    }
