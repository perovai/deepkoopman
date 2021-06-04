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

    def print_step(self, losses="", mode="train"):
        if mode == "train":
            n = self.n_train
        elif mode == "val":
            n = self.n_val
        elif mode == "test":
            n = self.n_test

        losses_str = (
            " | ".join("{}: {:.5f}".format(k, float(v)) for k, v in losses.items())
            if losses
            else ""
        )
        print(
            "Step {:>5}: Batch {:>3}/{} Epoch {:>5}/{} | ".format(
                self.global_step, self.batch_id + 1, n, self.epoch_id + 1, self.n_epochs
            )
            + losses_str,
            end="\r",
        )
