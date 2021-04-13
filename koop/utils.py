import torch.nn.init as init
from yaml import safe_load
from pathlib import Path


def load_params(path="./params.yaml", task="task1"):
    p = Path(path).expanduser().resolve()
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

    return params


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
