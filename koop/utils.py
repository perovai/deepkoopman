import uuid
from os.path import expandvars
from pathlib import Path

import torch.nn.init as init
from addict import Dict
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
    while path.exists():
        path = path.parent / (path.name + "-" + str(uuid.uuid4()).split("-")[0])
    if not dev:
        path.mkdir(exist_ok=False, parents=True)
    else:
        print("Dev mode: output directory is not created")
    print("Using output directory:", str(path))
    return str(path)


def init_weights(m, w_dist, b_dist, scale):
    classname = m.__class__.__name__
    if classname.find("BatchNorm2d") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            init.normal_(m.weight.data, 1.0, scale)
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    elif hasattr(m, "weight"):
        if w_dist == "normal":
            init.normal_(m.weight.data, 0.0, scale)
        elif w_dist == "xavier":
            init.xavier_normal_(m.weight.data, gain=scale)
        elif w_dist == "xavier_uniform":
            init.xavier_uniform_(m.weight.data, gain=1.0)
        elif w_dist == "kaiming":
            init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif w_dist == "orthogonal":
            init.orthogonal_(m.weight.data, gain=scale)
        elif w_dist == "none":  # uses pytorch's default init method
            m.reset_parameters()
        else:
            raise NotImplementedError(
                "initialization method [%s] is not implemented" % w_dist
            )
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
