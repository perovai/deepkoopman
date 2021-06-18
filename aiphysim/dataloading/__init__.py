from pathlib import Path


from torch.utils.data import DataLoader

from .koopman_dataset import KoopmanDataset
from .density_dataset import DensityDataset


def create_datasets(
    path, sequence_length, dataset_type="koopman", train_lim=-1, val_lim=-1, test_lim=-1
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

    if dataset_type == "density":
        train_files = list(Path(path).glob("train_*.h5"))
        val_files = list(Path(path).glob("val_*.h5"))

        return {
            "train": DensityDataset(train_files, train_lim),
            "val": DensityDataset(val_files, val_lim),
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

    path = Path(path).expanduser().resolve()
    datasets = create_datasets(
        path, sequence_length, dataset_type, lims["train"], lims["val"], lims["test"]
    )
    return {
        k: DataLoader(
            v, batch_size, shuffle=k == "train", pin_memory=True, num_workers=workers
        )
        for k, v in datasets.items()
    }
