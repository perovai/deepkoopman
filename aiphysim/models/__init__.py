from .koopman import DeepKoopman


def create_model(opts):
    if opts.model_type == "koopman":
        return DeepKoopman(opts)

    raise ValueError("Unknown model_type to create model" + str(opts.model_type))
