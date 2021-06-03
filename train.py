import minydra

from koop.utils import load_opts
from koop.trainer import Trainer

if __name__ == "__main__":

    args = minydra.Parser().args
    if args:
        args.pretty_print()

    opts = load_opts(
        args.get("yaml", "./config/opts.yaml"), args.get("task", "discrete")
    )

    trainer = Trainer(opts)
    trainer.setup()
    trainer.train()
