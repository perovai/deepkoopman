import sys
import ast
from pathlib import Path
import h5py
import numpy as np
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))
import minydra

from koop.utils import resolve, new_unique_path


def dat_to_array(fname, shape=3):
    with resolve(fname).open("r") as f:
        lines = f.readlines()
    values = [list(map(ast.literal_eval, line.strip().split())) for line in lines]
    matrix = [
        [v for value_tuple in tuple_list for v in value_tuple] for tuple_list in values
    ]
    array = np.array(matrix)

    return np.reshape(array, (-1, shape, array.shape[-1]))


def label_file(file_path, delay_fs=0.2):

    delay_path = file_path.parent
    t3_path = delay_path.parent
    intensity_path = t3_path.parent
    dataset_path = intensity_path.parent

    delay_idx = int(delay_path.name)
    t3_value = float(t3_path.name.split("_t2_")[-1])
    intensity = float(intensity_path.name.split("spectrum_Intensity_")[-1])
    dataset = dataset_path.name

    return {
        "delay": (delay_idx - 1) * delay_fs,
        "t3": t3_value,
        "i": intensity,
        "set": dataset,
    }


def sample_files(path, datasets, i1, t3, ignore_delays):
    """
    Create a list of all paths to DYNAMICSdat files as per the
    training sets (datasets), intensity (i1) and t3 delays found in path.

    within those folders (path/dataset/intensity/t3) there are 500 files
    named 1 to 500. `ignore_delays` is going to select every nth files
    like:

    keep_indices = np.arange(1, 501, ignore_delays)

    eg file path:
    /network/tmp1/schmidtv/perovai/training_set1/spectrum_Intensity_.02104/spectrum_Intensity_.02104_t2_20.0/481/DYNAMICSdat noqa: E501
    <------------ path ----------><-- dataset -><---- Intensity 02104 ---><-------------  t3 20 -----------><idx><--file-->
    Args:
        path (Path or str): base path for the datasets
        datasets (list(str)): The datasets to explore
        i1 (list(str)): Intensity values to select
        t3 (list(str)): The t3 delays to select
        ignore_delays (int): the step of the range to select files
    """

    if datasets == "all":
        datasets = [d for d in path.iterdir() if d.is_dir()]
    else:
        if isinstance(datasets, str):
            datasets = [datasets]
        datasets = [resolve(path / d) for d in datasets]
        ignoring = [d for d in datasets if not d.exists()]
        if ignoring:
            print(
                "Warning! Those datasets do not exist:\n"
                + "\n".join(list(map(str, ignoring)))
            )

    if i1 == "all":
        i1s = [resolve(i) for d in datasets for i in d.glob("spectrum_Intensity_*")]
    else:
        if not isinstance(i1, list):
            i1 = [i1]
        i1s = [
            resolve(t)
            for d in datasets
            for i in i1
            for t in d.glob(f"spectrum_Intensity_*{i}*")
        ]

    if t3 == "all":
        t3s = [resolve(i) for i1 in i1s for i in i1.glob("spectrum_Intensity_*")]
    else:
        if not isinstance(t3, list):
            t3 = [t3]
        t3s = [
            resolve(t)
            for i1 in i1s
            for i in t3
            for t in i1.glob(f"spectrum_Intensity_*_t2_{i}.0")
        ]

    keep_indices = np.arange(1, 501, ignore_delays)

    files = [
        resolve(t3 / str(delay) / "DYNAMICSdat") for t3 in t3s for delay in keep_indices
    ]

    return files


if __name__ == "__main__":
    parser = minydra.Parser()
    args = parser.args.resolve()

    # -----------------------------
    # -----  PARSE ARGUMENTS  -----
    # -----------------------------

    if "path" in args:
        path = resolve(args.path)
        assert path.exists()
    else:
        raise ValueError("Provide a base path: `path=X`")

    if "datasets" in args:
        datasets = args.datasets
    else:
        datasets = "all"

    if "i1" in args:
        i1 = args.i1
    else:
        i1 = "all"

    if "t3" in args:
        t3 = args.t3
    else:
        t3 = "all"

    if "ignore_delays" in args:
        ignore_delays = args.ignore_delays
    else:
        if "y" not in input(
            "No `ignore_delays` was provided. All 500 delays will be used. "
            + "Ok? [y/n] "
        ):
            print("Aborting")
            sys.exit()

    print("\n" + "-" * 50 + "\n")

    out = None
    if "out" not in args:
        if "y" not in input(
            ">> WARNING `out=X` is not provided. Using $SLURM_TMPDIR. Ok? [y/n]"
        ):
            print("Aborting")
            sys.exit()
        slurm_tmpdir = resolve("$SLURM_TMPDIR")
        assert slurm_tmpdir.exists()
        out = new_unique_path(slurm_tmpdir / "mini-dataset.h5")
    else:
        out = resolve(args.out)
        if out.is_dir():
            out = new_unique_path(out / "mini-dataset.h5")
            print(f"`out` is a directory, using {str(out)}")
        else:
            if out.exists():
                out = new_unique_path(out)
                print(f"Warning: outfile {out.name} exists, using {str(out)}")
            else:
                print(f"Creating dataset: {str(out)}")

    # ----------------------------
    # -----  CREATE DATASET  -----
    # ----------------------------
    files = sample_files(path, datasets, i1, t3, ignore_delays)

    with h5py.File(str(out), "w-") as f5:

        f5.attrs.update(dict(args))

        for i, f in enumerate(files):
            print(str(f[-30:]), str(i).zfill(4) + f"/{len(files)}", end="\r")
            labels = label_file(f)
            data = dat_to_array(f, shape=3)
            d = f5.create_dataset(
                f"trajectory_{i}",
                data=data,
                dtype="f32",
                compression="gzip",
                compression_opts=4,
            )
            d.attrs.update(labels)

    print(f"\nDone! Data is in {str(out)}")
