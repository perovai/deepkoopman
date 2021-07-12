import ast
import os
from os.path import expandvars
from pathlib import Path

import numpy as np
import torch
import yaml
from addict import Dict
from comet_ml import Experiment
from funkybob import RandomNameGenerator
from yaml import safe_load

from aiphysim.opts import Opts

COMET_KWARGS = {
    "auto_metric_logging": False,
    "parse_args": True,
    "log_env_gpu": True,
    "log_env_cpu": True,
    "display_summary_level": 0,
}

KNOWN_TASKS = set(["discrete", "pendulum", "fluidbox", "attractor"])


def mem_size(model):
    """
    Get model size in GB (as str: "N GB")
    """
    mem_params = sum(
        [param.nelement() * param.element_size() for param in model.parameters()]
    )
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs
    return f"{mem / 1e9:.4f} GB"


def num_params(model):
    """
    Print number of parameters in model's named children
    and total
    """
    s = "Number of parameters:\n"
    n_params = 0
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        s += f"  • {name:<15}: {n}\n"
        n_params += n
    s += f"{'total':<19}: {n_params}"

    return s


def resolve(path):
    """
    fully resolve a path:
    resolve env vars ($HOME etc.) -> expand user (~) -> make absolute

    Returns:
        pathlib.Path: resolved absolute path
    """
    return Path(expandvars(str(path))).expanduser().resolve()


def load_opts(path="./config/opts.yaml", task=None, known_tasks=KNOWN_TASKS):
    """
    Load opts from a yaml config for a specific task

    Returns:
        aiphysim.Opts: dot-accessible dict
    """
    p = resolve(path)
    print("Loading parameters from {}".format(str(p)))
    with p.open("r") as f:
        all_params = safe_load(f)

    if task is None:
        if "task" not in all_params:
            raise ValueError("No task provided or in the opts yaml file")
        task = all_params["task"]
    else:
        all_params["task"] = task

    params = {}
    for key, value in all_params.items():
        if isinstance(value, dict):
            if any(k in known_tasks for k in value):
                if task in value:
                    params[key] = value[task]
            else:
                params[key] = value
        else:
            params[key] = value

    return Opts(params)


def new_unique_path(path):
    """
    generates a new path from the input one by adding a random
    `adjective-noun` suffix.

    eg:

    new_unique_path("~/hello.txt")
        -> /Users/victor/hello.txt if it does not already exist
        -> /Users/victor/hello-gifted-boyed.txt if it does

    Works similarly for dirs

    Args:
        path (pathlike): path to get uniquely modified if it exists

    Returns:
        pathlib.Path: new non-existing path based on the input's path name and parent
    """
    path = resolve(path)
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix

    funky_gen = iter(RandomNameGenerator(members=2, separator="-"))

    while path.exists():
        funky = next(funky_gen)
        path = path.parent / (stem + "-" + funky + suffix)

    return path


def make_output_dir(path, dev=False):
    """
    Create output dir with the path's name.
    If it exists, will append random `-adjective-name`
    suffix to make it uniquely identifiable

    mkdir will not be called if `dev` is True

    Returns:
        pathlib.Path: path to a unique empty dir
    """
    path = new_unique_path(path)

    if not dev:
        path.mkdir(exist_ok=False, parents=True)
    else:
        print("Dev mode: output directory is not created")

    print("Using output directory:", str(path))

    return path


def get_optimizer(opts, model):
    """
    Create optimizer and scheduler according to the opts
    """
    opt_name = opts.optimizer.name.lower()
    if opt_name == "adam":
        opt_class = torch.optim.Adam
    else:
        raise NotImplementedError("Unknown optimizer " + opts.optimizer.name)
    optimizer = opt_class(model.parameters(), lr=opts.optimizer.lr)

    scheduler_name = opts.optimizer.get("scheduler", "").lower()
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=20,
            min_lr=1e-6,
        )
    else:
        scheduler = None

    return optimizer, scheduler


def upload_code_and_parameters(exp: Experiment, opts: Opts):
    # code
    py_files = []
    py_files += list(Path(__file__).resolve().parent.parent.glob("./*.py"))
    py_files += list(Path(__file__).resolve().parent.parent.glob("./scripts/*.py"))
    py_files += list(Path(__file__).resolve().parent.parent.glob("./aiphysim/*.py"))
    for py in py_files:
        exp.log_asset(str(py), file_name=f"{py.parent.name}/{py.name}")

    # parameters
    opts_dict = flatten_opts(opts)
    opts_dict["output_path"] = str(opts_dict["output_path"])
    exp.log_parameters(opts_dict, prefix="opts")


def save_config(opts, exp=None):
    if opts.get("dev"):
        print("Dev mode : config not saved to disk")
        return
    output_path = opts.output_path
    to_save = opts.to_dict()
    to_save["output_path"] = str(to_save["output_path"])
    with open(output_path / "opts.yaml", "w") as f:
        yaml.safe_dump(to_save, f)

    if exp is not None:
        with open(output_path / "comet_url.txt", "w") as f:
            f.write(exp.url)


def find_existing_comet_id(path):
    comet_file = path / "comet_url.txt"
    assert comet_file.exists()

    with comet_file.open("r") as f:
        comet_url = f.read().strip()

    return comet_url.split("/")[-1]


def flatten_opts(opts: Dict) -> dict:
    """Flattens a multi-level addict.Dict or native dictionnary into a single
    level native dict with string keys representing the keys sequence to reach
    a value in the original argument.

    d = addict.Dict()
    d.a.b.c = 2
    d.a.b.d = 3
    d.a.e = 4
    d.f = 5
    flatten_opts(d)
    >>> {
        "a.b.c": 2,
        "a.b.d": 3,
        "a.e": 4,
        "f": 5,
    }

    Args:
        opts (addict.Dict or dict): addict dictionnary to flatten

    Returns:
        dict: flattened dictionnary
    """
    values_list = []

    def p(d, prefix="", vals=None):
        if vals is None:
            vals = []
        for k, v in d.items():
            if isinstance(v, (Dict, dict)):
                p(v, prefix + k + ".", vals)
            elif isinstance(v, list):
                if v and isinstance(v[0], (Dict, dict)):
                    for i, m in enumerate(v):
                        p(m, prefix + k + "." + str(i) + ".", vals)
                else:
                    vals.append((prefix + k, str(v)))
            else:
                if isinstance(v, Path):
                    v = str(v)
                vals.append((prefix + k, v))

    p(opts, vals=values_list)
    return dict(values_list)


def clean_checkpoints(path, n_max=5):
    path = resolve(path)
    ckpts = list(path.glob("*.ckpt"))
    ckpts = [c for c in ckpts if c.name not in ["latest.ckpt", "best.ckpt"]]

    if len(ckpts) < n_max:
        return

    sorted_ckpts = sorted(ckpts, key=lambda c: float(c.stem.split("loss_")[-1]))
    os.remove(sorted_ckpts[-1])


def dat_to_array(fname, shape=3):
    with resolve(fname).open("r") as f:
        lines = f.readlines()
    values = [list(map(ast.literal_eval, line.strip().split())) for line in lines]
    matrix = [
        [v for value_tuple in tuple_list for v in value_tuple] for tuple_list in values
    ]
    array = np.array(matrix)

    return np.reshape(array, (-1, shape, array.shape[-1]))
