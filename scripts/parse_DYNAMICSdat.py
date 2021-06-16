import sys
import ast
from pathlib import Path
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

    if "file" in args:
        files = [resolve(args.file)]
    elif "files" in args:
        files = [resolve(p) for p in args.files]
    elif "glob" in args:
        files = [resolve(p) for p in Path.glob(args.glob)]
    else:
        raise ValueError('Provide file=path or glob=path/*.X or files=\'["a", "b"]\'')

    for f in files:
        array = dat_to_array(f)
