import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

sys.path.append("../")
from aiphysim.opts import Opts
from aiphysim.trainer import Trainer
from aiphysim.utils import load_opts, make_output_dir, resolve

opts = load_opts("../config/opts.yaml", "discrete")
opts.output_path = make_output_dir(opts.output_path, dev=False)
opts.data_folder = "../datasets/DiscreteSpectrumExample"
trainer = Trainer(opts)

trainer.setup()

batch = trainer.dev_batch()
trainer.model(batch)
