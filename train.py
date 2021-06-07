import minydra
from comet_ml import Experiment

from koop.opts import Opts
from koop.trainer import Trainer
from koop.utils import (
    comet_kwargs,
    load_opts,
    make_output_dir,
    save_config,
    upload_code,
)

exp = None

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
    opts.output_path = make_output_dir(opts.output_path, dev=opts.get("dev"))

    if opts.comet.use:
        exp = Experiment(
            workspace=opts.comet.workspace,
            project_name=opts.comet.project_name,
            **comet_kwargs
        )
        exp.set_name(opts.output_path.name)
        upload_code(exp)

    save_config(opts, exp)

    trainer = Trainer(opts, exp)
    trainer.setup()
    trainer.train()
    print("\n >>>> Done training model in", str(opts.output_path))
