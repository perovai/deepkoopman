import json
import sys
from pathlib import Path

from minydra import Parser
import numpy as np
from tqdm import tqdm

from subsample_density_dataset import label_file

if __name__ == "__main__":

    parser = Parser()
    args = parser.args.resolve()

    base = Path("/network/tmp1/schmidtv/perovai")
    out = Path("/network/tmp1/schmidtv/perovai/labeled_data_paths.json")
    train_split = 0.8

    if "base" in args:
        base = Path(args.base).expanduser().resolve()
    if "out" in args:
        out = Path(args.out).expanduser().resolve()
        if out.exists():
            print(str(out), "already exists")
            if "y" not in input("Continue? [y/n]"):
                sys.exit()
    if "train_split" in args:
        train_split = float(args.train_split)

    assert out.parent.exists()

    print("Using base discovery path", str(base))
    print("Using output json", str(out))

    fname = "DYNAMICSdat"

    data_files = [f for f in tqdm(base.glob(f"**/{fname}"))]

    perm = np.random.permutation(len(data_files))
    train_idx = int(train_split * len(data_files))

    train_jfiles = {
        str(data_files[k]): label_file(data_files[k]) for k in tqdm(perm[:train_idx])
    }
    val_jfiles = {
        str(data_files[k]): label_file(data_files[k]) for k in tqdm(perm[train_idx:])
    }

    train_out = out.parent / f"train_{out.name}"
    val_out = out.parent / f"val_{out.name}"

    with train_out.open("w") as f:
        json.dump(train_jfiles, f)
    with val_out.open("w") as f:
        json.dump(val_jfiles, f)
