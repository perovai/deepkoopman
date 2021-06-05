from os.path import expandvars
from pathlib import Path

import torch
import yaml
from comet_ml import Experiment
from funkybob import RandomNameGenerator
from yaml import safe_load

from koop.opts import Opts

comet_kwargs = {
    "auto_metric_logging": False,
    "parse_args": True,
    "log_env_gpu": True,
    "log_env_cpu": True,
    "display_summary_level": 0,
}


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


def load_opts(path="./config/opts.yaml", task=None):
    """
    Load opts from a yaml config for a specific task

    Returns:
        koop.Opts: dot-accessible dict
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
            if "all" in value:
                params[key] = value["all"]
            elif task in value:
                params[key] = value[task]
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


def upload_code(exp: Experiment):
    py_files = []
    py_files += list(Path(__file__).resolve().parent.parent.glob("./*.py"))
    py_files += list(Path(__file__).resolve().parent.parent.glob("./scripts/*.py"))
    py_files += list(Path(__file__).resolve().parent.parent.glob("./koop/*.py"))
    for py in py_files:
        exp.log_asset(str(py), file_name=f"{py.parent.name}/{py.name}")


def save_config(opts, exp):
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
