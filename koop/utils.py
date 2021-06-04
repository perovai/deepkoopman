from os.path import expandvars
from pathlib import Path

import torch
from funkybob import RandomNameGenerator
from yaml import safe_load

from koop.opts import Opts


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


def load_opts(path="./config/opts.yaml", task="discrete"):
    """
    Load opts from a yaml config for a specific task

    Returns:
        koop.Opts: dot-accessible dict
    """
    p = resolve(path)
    print("Loading parameters from {}".format(str(p)))
    with p.open("r") as f:
        all_params = safe_load(f)

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


def make_output_dir(path, dev=False):
    """
    Create output dir with the path's name.
    If it exists, will append random `-adjective-name`
    suffix to make it identifiable

    mkdir will not be called if `dev` is True

    Returns:
        pathlib.Path: path to a unique empty dir
    """
    path = resolve(path)
    funky_gen = iter(RandomNameGenerator(members=2, separator="-"))
    while path.exists():
        funky = next(funky_gen)
        path = path.parent / (path.name + "-" + funky)
    if not dev:
        path.mkdir(exist_ok=False, parents=True)
    else:
        print("Dev mode: output directory is not created")
    print("Using output directory:", str(path))
    return str(path)


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
