from .koopman import DeepKoopman
from .spacetime import SpaceTime


def create_model(opts):
    if opts.model_type == "koopman":
        return DeepKoopman(opts)
    if opts.model_type == "spacetime":
        return SpaceTime(opts)

    raise ValueError("Unknown model_type to create model" + str(opts.model_type))
