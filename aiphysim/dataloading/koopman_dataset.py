import numpy as np
import torch
from torch.utils.data import Dataset


class KoopmanDataset(Dataset):
    def __init__(self, file_paths, sequence_length=51, limit=-1):
        self.file_paths = file_paths
        self.sequence_length = sequence_length
        self.input_dim = None

        self.data = []
        for file in self.file_paths:
            self.data.append(self.load_file(file))
        self.data = torch.cat([n for n in self.data])
        if limit > 0:
            self.data = self.data[:limit]

    def load_file(self, file_path):
        x = torch.from_numpy(np.genfromtxt(file_path, delimiter=",", dtype=np.float32))
        if self.input_dim is None:
            self.input_dim = x.shape[1]

        return x.reshape(-1, self.sequence_length, self.input_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
