import minydra

from koop.trainer import Trainer
from koop.utils import Opts, load_opts

if __name__ == "__main__":

    args = minydra.Parser().args
    if args:
        args.pretty_print()
    args = Opts(args)

    opts = load_opts(
        args.get("yaml", "./config/opts.yaml"), args.get("task", "discrete")
    )
    opts.update(args)

    trainer = Trainer(opts)
    trainer.setup()
    trainer.train()
