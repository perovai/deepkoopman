import atexit

import minydra
from addict import Dict
from comet_ml import Experiment

from aiphysim.trainer import Trainer
from aiphysim.utils import (
    COMET_KWARGS,
    load_opts,
    make_output_dir,
    save_config,
    upload_code_and_parameters,
)

exp = None


def print_trainer_times(trainer):
    if hasattr(trainer, "_mean_times"):
        print("\nTrainer's average processing times:")
        for func_name, mean_time in trainer._mean_times.items():
            print(f"  {func_name:30}: {mean_time:.3f}s")
        print()
    else:
        print("No _mean_times to print, exiting.")


if __name__ == "__main__":

    parser = minydra.Parser()
    args = parser.args.resolve()

    if args:
        args.pretty_print()
    args = Dict(args)

    opts = load_opts(
        defaults=args.get("yaml", "./config/defaults.yaml"),
        task=args.get("task", "discrete"),
        task_yaml=args.get("task_yaml"),
    )
    opts.update(args)
    opts.output_path = make_output_dir(opts.output_path, dev=opts.get("dev"))

    if opts.comet.use:
        exp = Experiment(project_name=opts.comet.project_name, **COMET_KWARGS)
        exp.set_name(opts.output_path.name)
        upload_code_and_parameters(exp, opts)

    trainer = Trainer(opts, exp).setup()

    save_config(trainer.opts, exp)
    atexit.register(print_trainer_times, trainer)

    trainer.train()

    print("\n >>>> Done training model in", str(opts.output_path))
