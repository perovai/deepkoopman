from os.path import expandvars
from pathlib import Path

import torch
from addict import Dict
from funkybob import RandomNameGenerator
from yaml import safe_load


class Opts(Dict):
    def __missing__(self, key):
        raise KeyError(key)


def resolve(path):
    return Path(expandvars(str(path))).expanduser().resolve()


def load_opts(path="./config/opts.yaml", task="discrete"):
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
