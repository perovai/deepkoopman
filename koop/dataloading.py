from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class KoopmanDataset(Dataset):
    def __init__(self, file_paths, sequence_length=51):
        self.file_paths = file_paths
        self.sequence_length = sequence_length
        self.input_dim = None

        self.data = []
        for file in self.file_paths:
            self.data.append(self.load_file(file))
        self.data = torch.cat([n for n in self.data])

    def load_file(self, file_path):
        x = torch.from_numpy(np.genfromtxt(file_path, delimiter=",", dtype=np.float32))
        if self.input_dim is None:
            self.input_dim = x.shape[1]

        return x.reshape(-1, self.sequence_length, self.input_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def create_datasets(path, sequence_length):
    print("Creating datasets from ", str(path))
    train_files = list(Path(path).glob("*_train*.csv"))
    val_files = list(Path(path).glob("*_val*.csv"))
    test_files = list(Path(path).glob("*_test*.csv"))

    return {
        "train": KoopmanDataset(train_files, sequence_length),
        "val": KoopmanDataset(val_files, sequence_length),
        "test": KoopmanDataset(test_files, sequence_length),
    }


def create_dataloaders(path, sequence_length, batch_size):
    """
    Return a dict of data loaders with keys train/val/test

    Args:
        path (pathlike): Path to read the data from
        sequence_length (int): How to reshape the data from N x dim to B x seq x dim
        batch_size (int): Batch size

    Returns:
        dict: {"train": dataloader, "val": dataloader, "test": dataloader}
    """
    path = Path(path).expanduser().resolve()
    datasets = create_datasets(path, sequence_length)
    return {
        k: DataLoader(v, batch_size, shuffle=k == "train") for k, v in datasets.items()
    }
