import json
import sys
from pathlib import Path

from minydra import Parser
from tqdm import tqdm

from .subsample_density_dataset import label_file

if __name__ == "__main__":

    parser = Parser()
    args = parser.parse_args().resolve()

    base = Path("/network/tmp1/schmidtv/perovai")
    out = Path("/network/tmp1/schmidtv/perovai/labeled_data_paths.json")

    if "base" in args:
        base = Path(args.base).expanduser().resolve()
    if "out" in args:
        out = Path(args.out).expanduser().resolve()
        if out.exists():
            print(str(out), "already exists")
            if "y" not in input("Continue? [y/n]"):
                sys.exit()

    assert out.parent.exists()

    print("Using base discovery path", str(base))
    print("Using output json", str(out))

    fname = "DYNAMICSdat"

    data_files = [f for f in tqdm(base.glob(f"**/{fname}"))]

    jfiles = {f: label_file(f) for f in tqdm(data_files)}

    with out.open("w") as f:
        json.dump(jfiles, out)
