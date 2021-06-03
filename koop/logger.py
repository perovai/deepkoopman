class Logger:
    def __init__(self, opts=None, exp=None):
        self.opts = opts
        self.exp = exp
        self.global_step = 0

    def print_step(self, losses=""):
        losses_str = (
            " | ".join("{}: {:.4f}".format(k, float(v)) for k, v in losses.items())
            if losses
            else ""
        )
        print(
            "Step {:<5}: Batch {:<3} Epoch {:<5}".format(
                self.global_step, self.batch_id, self.epoch_id
            )
            + losses_str,
            end="\r",
        )
