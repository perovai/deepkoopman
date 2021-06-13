import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from koop.opts import Opts
from koop.trainer import Trainer

from koop.utils import (
    load_opts,
    make_output_dir,
    resolve
)


opts = load_opts("../config/opts.yaml",  "discrete")
opts.output_path = make_output_dir(opts.output_path, dev=False)
opts.data_folder = "../datasets/DiscreteSpectrumExample"
trainer = Trainer(opts)

trainer.setup()

batch = trainer.dev_batch()
trainer.model(batch)
