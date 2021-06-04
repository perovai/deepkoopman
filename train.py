import minydra

from koop.opts import Opts
from koop.trainer import Trainer
from koop.utils import load_opts

if __name__ == "__main__":

    parser = minydra.Parser()
    args = parser.args.resolve()

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
