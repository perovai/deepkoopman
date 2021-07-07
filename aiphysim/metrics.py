import traceback as tb

import torch


class BaseMetrics:
    def __init__(self, opts):
        self.metrics_funcs = {
            loss_name: getattr(BaseMetrics, loss_name) for loss_name in opts.losses
        }

    @staticmethod
    def mse(targets, predictions):
        return torch.nn.functional.mse_loss(predictions, targets)

    @staticmethod
    def time_mse(targets, predictions):
        """
        Assumes batch x time x [...others]

        Returns average mse per time step across batch and other dimensions
        """
        mse = torch.nn.functional.mse_loss(predictions, targets, reduction="none")
        return torch.transpose(mse, 1, 0).reshape(mse.shape[1], -1).mean(-1)

    @torch.no_grad()
    def compute(self, *args):
        self.set_args(*args)

        metrics = {}

        for name, metric in self.metrics_funcs.items():
            if name in self.args:
                try:
                    metrics[name] = metric(*self.args[name])
                except Exception as e:
                    print("Error in metric", name, "\n", tb.format_exc())
                    raise Exception(e)
        return metrics


class SpaceTimeMetrics(BaseMetrics):
    def set_args(self, inputs, predictions, model):
        embedding, y, latent_evol = predictions

        self.args = {
            "mse": (inputs[0, :, :], y[0]),
            "time_mse": (inputs, y),
        }


def get_metric(opts):
    metric_type = opts.get("metric_type")
    if metric_type == "spacetime":
        return SpaceTimeMetrics(opts)

    raise ValueError("Unknown loss type: " + str(metric_type))
