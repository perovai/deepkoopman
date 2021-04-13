import torch
from koop.model import AutoEncoder
from koop.utils import load_params

if __name__ == "__main__":

    params = load_params("./params.yaml", "task1")

    model = AutoEncoder(
        params["widths"], params["dist_w"], params["dist_b"], params["scale"]
    )

    dummy_data = torch.rand(10, params["widths"][0])

    y = model(dummy_data)