import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from aiphysim.utils import new_unique_path  # noqa: E402

if __name__ == "__main__":

    tasks = ["density", "spacetime"]

    params = {
        "epochs": 1,
        "batch_size": 32,
        "workers": 2,
        "limit.train": 64,
        "limit.val": 64,
        "comet.use": False,
    }

    outfile = new_unique_path(
        Path(__file__).resolve().parent.parent / "runs" / "out_test_all.txt"
    )

    out = f" >> {outfile}"

    for task in tasks:
        params["task"] = task
        print("Executing:", params)

        cd = "cd .. && "
        train = "python train.py " + " ".join(f"{k}={v}" for k, v in params.items())
        os.system(cd + train + out)
