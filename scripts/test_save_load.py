import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from aiphysim.trainer import Trainer  # noqa: E402
from aiphysim.utils import load_opts, make_output_dir, save_config  # noqa: E402

if __name__ == "__main__":

    pwd = Path().resolve()

    if pwd.name != "deepkoopman" and "aiphysim" not in [
        p.name for p in pwd.iterdir() if p.is_dir()
    ]:
        os.chdir(pwd.parent)

    opts = load_opts(
        Path(__file__).parent.parent / "config" / "opts.yaml", task="pendulum"
    )
    opts.output_path = make_output_dir(opts.output_path, dev=opts.get("dev"))
    opts.limit.train = 16
    opts.limit.val = 16
    opts.batch_size = 4
    opts.epochs = 2
    opts.comet.use = False

    print(opts)

    trainer = Trainer(opts).setup()
    save_config(trainer.opts)
    trainer.train()

    batch = trainer.dev_batch()

    preds_0 = trainer.model(batch)

    trainer_resumed = Trainer.resume_from_path(
        trainer.opts.output_path, exp_type=None, inference_only=True
    )

    preds_1 = trainer_resumed.model(batch)

    for p0, p1 in zip(preds_0, preds_1):
        assert (p0 == p1).all()

    print("Success: inferences match")
