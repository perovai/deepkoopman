from koop.dataloading import create_dataloaders
from koop.utils import load_params
from koop.model import DeepKoopman
import minydra

if __name__ == "__main__":

    args = minydra.Parser().args
    if args:
        args.pretty_print()

    params = load_params(
        args.get("yaml", "./config/params.yaml"), args.get("task", "discrete")
    )

    dataloaders = create_dataloaders(
        params["data"], params["sequence_length"], params["batch_size"]
    )

    params["input_dim"] = dataloaders["train"].dataset.input_dim

    model = DeepKoopman(params)
