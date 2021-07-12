import sys

sys.path.append("../")
from aiphysim.trainer import Trainer  # noqa: E402
from aiphysim.utils import load_opts, make_output_dir  # noqa: E402

opts = load_opts("../config/opts.yaml", "discrete")
opts.output_path = make_output_dir(opts.output_path, dev=False)
opts.data_folder = "../datasets/DiscreteSpectrumExample"
trainer = Trainer(opts)

trainer.setup()

batch = trainer.dev_batch()
trainer.model(batch)
