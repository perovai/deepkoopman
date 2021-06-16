import sys
import ast
from pathlib import Path
import glob
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import minydra

from koop.utils import resolve


def dat_to_array(fname):
    with resolve(fname).open("r") as f:
        lines = f.readlines()
    values = [list(map(ast.literal_eval, line.strip().split())) for line in lines]
    matrix = [
        [v for value_tuple in tuple_list for v in value_tuple] for tuple_list in values
    ]
    return np.array(matrix)


if __name__ == "__main__":
    parser = minydra.Parser()
    args = parser.args.resolve()

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
        if isinstance(i1, str):
            i1 = [i1]
        i1s = [resolve(d / f"spectrum_Intensity_.{i}") for d in datasets for i in i1]

    if t3 == "all":
        t3s = [resolve(i) for i1 in i1s for i in i1.glob("spectrum_Intensity_*")]
    else:
        if isinstance(t3, str):
            t3 = [t3]
        t3s = [
            resolve(t)
            for i1 in i1s
            for i in t3
            for t in i1.glob(f"spectrum_Intensity_*_t2_{i}.0")
        ]

    delays = np.arange(1, 501, ignore_delays)

    files = [resolve(t3 / str(delay) / "DYNAMICSdat") for t3 in t3s for delay in delays]
