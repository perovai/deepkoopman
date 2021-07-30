import traceback as tb

import numpy as np
import torch

from aiphysim.utils import timeit


class BaseLoss:
    def __init__(self, opts, is_loss=True):
        self.weights = (
            {name: float(weight) for name, weight in opts.weights.items()}
            if is_loss
            else {}
        )

        self.is_loss = is_loss

    @staticmethod
    def mse(reconstruction, state):
        return torch.nn.functional.mse_loss(reconstruction, state)

    @staticmethod
    def prediction(inputs, y, shifts):
        pred_loss = torch.tensor(0.0)
        for shift in shifts:
            pred_loss += torch.nn.functional.mse_loss(inputs[shift, :, :], y[shift])
        pred_loss /= len(shifts)

        return pred_loss

    @staticmethod
    def linear(embedding, latent_evol, middle_shifts):
        lin_loss = torch.tensor(0.0)
        for j, shift in enumerate(middle_shifts):
            lin_loss += torch.nn.functional.mse_loss(
                embedding[shift, :, :], latent_evol[j]
            )

        lin_loss /= len(middle_shifts)
        return lin_loss

    @staticmethod
    def latent_evolution(embedding_evolution, sequence, encoder):
        embedding_sequence = encoder(sequence)
        return torch.nn.functional.mse_loss(embedding_evolution, embedding_sequence)

    @staticmethod
    def inf_norm(inputs, y):
        inf = float("inf")
        Linf1_penalty = torch.norm(torch.norm(y[0] - inputs[0, :, :], p=inf), p=inf)
        Linf2_penalty = torch.norm(torch.norm(y[1] - inputs[1, :, :], p=inf), p=inf)
        return Linf1_penalty + Linf2_penalty

    @staticmethod
    def l2(model):
        l2_reg = 0
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param)
        return l2_reg

    @staticmethod
    def time_mse(targets, predictions):
        """
        Assumes batch x time x [...others]

        Returns average mse per time step across batch and other dimensions
        """
        mse = torch.nn.functional.mse_loss(predictions, targets, reduction="none")
        return torch.transpose(mse, 1, 0).reshape(mse.shape[1], -1).mean(-1)

    @timeit
    def compute(self, *args):
        self.set_args(*args)

        losses = {}

        if self.is_loss:
            losses["total"] = 0.0

        for loss_params in self.args:

            # args are
            #
            # (function_type, tuple(args))
            # or
            # (function_type, function_name, tuple(args))
            #
            # function_name is used to differentiate 2 identical losses
            # applied to different inputs

            if len(loss_params) == 3:
                loss_func, loss_name, loss_args = loss_params
            else:
                loss_func, loss_args = loss_params
                loss_name = loss_func

            if hasattr(self, loss_func):
                try:
                    weight = float(self.weights.get(loss_name, 1))
                    if weight <= 0:
                        continue

                    loss = getattr(self, loss_func)
                    losses[loss_name] = loss(*loss_args)
                    if self.is_loss:
                        losses["total"] += losses[loss_name] * weight

                except Exception as e:
                    print(
                        ">>> Error in loss",
                        loss_name,
                        "for func",
                        loss_func,
                        "with weight",
                        self.weights.get(loss_name, 1),
                    )
                    print("\n" + tb.format_exc())
                    raise Exception(e)
        return losses


class KoopmanLoss(BaseLoss):
    def __init__(self, opts, is_loss=False):
        super().__init__(opts, is_loss)

        self.shifts = np.arange(opts.num_shifts) + 1
        self.middle_shifts = np.arange(opts.num_shifts_middle) + 1

    def set_args(self, inputs, predictions, model):
        """
        Compute the weighted loss with respect to targets and predictions

        Args:
            targets (dict): dictionary of target values
            predictions (dict): dictionary of predicted values
        """
        embedding, y, latent_evol = predictions

        self.args = [
            ("mse", (inputs[0, :, :], y[0])),
            ("prediction", (inputs, y, self.shifts)),
            ("linear", (embedding, latent_evol, self.middle_shifts)),
            ("l2", (model,)),
            ("inf_norm", (inputs, y)),
        ]


class DensityLoss(BaseLoss):
    def set_args(self, batch, predictions, model):

        encoded_ts, decoded_ts, next_zs, next_decoded_ts = predictions

        time_series = batch["data"]

        self.args = [
            ("l2", (model,)),
            ("mse", "mse_latent", (encoded_ts[1:], next_zs[:-1])),
            ("mse", "mse_next_decoded", (time_series[1:], next_decoded_ts[:-1])),
            ("mse", "mse_decoded", (time_series, decoded_ts)),
        ]


class DensityMetrics(BaseLoss):
    def set_args(self, batch, predictions, model):

        encoded_ts, decoded_ts, next_zs, next_decoded_ts = predictions
        time_series = batch["data"]

        self.args = [
            ("l2", (model,)),
            ("mse", "mse_latent", (encoded_ts[1:], next_zs[:-1])),
            ("mse", "mse_next_decoded", (time_series[1:], next_decoded_ts[:-1])),
            ("mse", "mse_decoded", (time_series, decoded_ts)),
        ]


class SpaceTimeLoss(BaseLoss):
    def set_args(self, batch, predictions, model):

        encoded_ts, decoded_ts, next_zs, next_decoded_ts = predictions

        time_series = batch["data"]

        self.args = [
            ("l2", (model,)),
        ]


class SpaceTimeMetrics(BaseLoss):
    def set_args(self, batch, predictions, model):

        encoded_ts, decoded_ts, next_zs, next_decoded_ts = predictions

        time_series = batch["data"]

        self.args = [
            ("l2", (model,)),
        ]


class Unet3DLoss(BaseLoss):
    def set_args(self, inputs, predictions, model):
        """
        Compute the weighted loss with respect to targets and predictions

        Args:
            inputs (dict): dictionary of target values
            predictions (dict): dictionary of predicted values
        """

        self.args = [("mse", "reconstruction", (inputs, predictions))]


class Unet3DMetric(BaseLoss):
    def set_args(self, inputs, predictions, model):
        """
        Compute the weighted loss with respect to targets and predictions

        Args:
            inputs (dict): dictionary of target values
            predictions (dict): dictionary of predicted values
        """

        self.args = [("mse", "reconstruction", (inputs, predictions))]


def get_loss_and_metrics(opts):
    loss_type = opts.get("loss_type")
    metrics_type = opts.get("metrics_type")

    loss = metrics = None

    if loss_type == "koopman":
        loss = KoopmanLoss(opts)
    elif loss_type == "spacetime":
        loss = SpaceTimeLoss(opts)
    elif loss_type == "density":
        loss = DensityLoss(opts)
    elif loss_type == "3dunet":
        loss = Unet3DLoss(opts)
    else:
        raise ValueError("Unknown loss type: " + str(loss_type))

    if metrics_type == "spacetime":
        metrics = SpaceTimeMetrics(opts, False)
    elif metrics_type == "density":
        metrics = DensityMetrics(opts, False)
    elif metrics_type == "3dunet":
        metrics = Unet3DMetric(opts, False)
    else:
        raise ValueError("Unknown metrics type: " + str(metrics_type))

    return loss, metrics
