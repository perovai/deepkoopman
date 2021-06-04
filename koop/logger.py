from queue import deque
from time import time
from datetime import datetime
import numpy as np


class Logger:
    def __init__(self, opts=None, exp=None, n_train=None, n_val=None, n_test=None):
        self.opts = opts
        self.exp = exp
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test

        self.global_step = 0
        self.batch_id = 0
        self.epoch_id = 0

        self.n_epochs = self.opts.get("epochs")

        self.qs = {
            "train": deque([], maxlen=25),
            "val": deque([], maxlen=25),
            "test": deque([], maxlen=25),
        }

        self.ts = {
            "train": time(),
            "val": time(),
            "test": time(),
        }

    def now(self):
        return str(datetime.now()).split(".")[0].split()[-1]

    def print_step(self, losses="", mode="train"):
        if mode == "train":
            n = self.n_train
        elif mode == "val":
            n = self.n_val
        elif mode == "test":
            n = self.n_test

        now = self.now()

        nt = time()
        diff = nt - self.ts[mode]
        self.ts[mode] = nt
        self.qs[mode].append(diff)
        t = f"{np.mean(self.qs[mode]): .2f}s/b"

        losses_str = (
            " | ".join("{}: {:.5f}".format(k, float(v)) for k, v in losses.items())
            if losses
            else ""
        )
        print(
            "[{} {}] {:>5} {:>3}/{} {:>3}/{} | ".format(
                now,
                t,
                self.global_step,
                self.batch_id + 1,
                n,
                self.epoch_id + 1,
                self.n_epochs,
            )
            + losses_str,
            end="\r",
        )
