from .koopman import DeepKoopman
from .spacetime import SpaceTime
from .unet import UNet3d


def create_model(opts):
    if opts.model_type == "koopman":
        return DeepKoopman(opts)
    if opts.model_type == "spacetime":
        return SpaceTime(opts)
    if opts.model_type == "3dunet":
        return UNet3d(opts)

    raise ValueError("Unknown model_type to create model" + str(opts.model_type))
