import json

import h5py
import torch
from torch.utils.data import Dataset

from aiphysim.utils import dat_to_array


class DatDensityDataset(Dataset):
    def __init__(self, json_file, limit=-1) -> None:
        super().__init__()

        with open(json_file, "r") as f:
            self.paths = json.load(f)

        self.keys = list(self.paths.keys())

        if limit > 0:
            self.keys = self.keys[:limit]
            self.paths = {k: self.paths[k] for k in self.keys}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return dat_to_array(self.paths[self.keys[index]])


class H5DensityDataset(Dataset):
    def __init__(self, h5_paths, limit=-1):
        self.limit = limit
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self.indices = {}
        self.input_dim = None
        idx = 0
        for a, archive in enumerate(self.archives):
            if self.input_dim is None:
                self.input_dim = list(archive.values())[0].shape[1:]
            for i in range(len(archive)):
                self.indices[idx] = (a, i)
                idx += 1

        self._archives = None

    @property
    def archives(self):
        if self._archives is None:
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __getitem__(self, index):
        a, i = self.indices[index]
        archive = self.archives[a]
        dataset = archive[f"trajectory_{i}"]
        data = torch.from_numpy(dataset[:])
        labels = dict(dataset.attrs)

        return {"data": data, "labels": labels}

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)
