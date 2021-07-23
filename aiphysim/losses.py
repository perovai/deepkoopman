import traceback as tb

import numpy as np
import torch


class BaseLoss:
    def __init__(self, opts, is_loss=False):
        self.weights = (
            {name: float(weight) for name, weight in opts.weights.items()}
            if is_loss
            else None
        )
        self.loss_funcs = {
            loss_name: getattr(BaseLoss, loss_name)
            for loss_name in (opts.losses if is_loss else opts.metrics)
        }
        self.is_loss = is_loss

        self.shifts = np.arange(opts.num_shifts) + 1
        self.middle_shifts = np.arange(opts.num_shifts_middle) + 1

    @staticmethod
    def reconstruction(reconstruction, state):
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
        l2_reg = torch.tensor(0.0)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        return l2_reg

    @staticmethod
    def time_mse(targets, predictions):
        """
        Assumes batch x time x [...others]

        Returns average mse per time step across batch and other dimensions
        """
        mse = torch.nn.functional.mse_loss(predictions, targets, reduction="none")
        return torch.transpose(mse, 1, 0).reshape(mse.shape[1], -1).mean(-1)

    def compute(self, *args):
        self.set_args(*args)

        losses = {}

        if self.is_loss:
            losses["total"] = 0.0

        for name, loss in self.loss_funcs.items():
            if name in self.args:
                try:
                    losses[name] = loss(*self.args[name])
                    if self.is_loss:
                        weight = float(self.weights.get(name, 1))
                        losses["total"] += losses[name] * weight
                except Exception as e:
                    print(
                        "Error in loss", name, "with weight", self.weights.get(name, 1)
                    )
                    print("\n" + tb.format_exc())
                    raise Exception(e)
        return losses


class KoopmanLoss(BaseLoss):
    def set_args(self, inputs, predictions, model):
        """
        Compute the weighted loss with respect to targets and predictions

        Args:
            targets (dict): dictionnary of target values
            predictions (dict): dictionnary of predicted values
        """
        embedding, y, latent_evol = predictions

        self.args = {
            "reconstruction": (inputs[0, :, :], y[0]),
            "prediction": (inputs, y, self.shifts),
            "linear": (embedding, latent_evol, self.middle_shifts),
            "l2": (model,),
            "inf_norm": (inputs, y),
        }


class SpaceTimeLoss(BaseLoss):
    def set_args(self, inputs, predictions, model):

        self.args = {
            # TODO
        }


class SpaceTimeMetrics(BaseLoss):
    def set_args(self, inputs, predictions, model):

        self.args = {
            # TODO
        }


class Unet3DLoss(BaseLoss):
    def set_args(self, inputs, predictions, model):
        """
        Compute the weighted loss with respect to targets and predictions

        Args:
            inputs (dict): dictionnary of target values
            predictions (dict): dictionnary of predicted values
        """

        self.args = {"reconstruction": (inputs, predictions)}


def get_loss_and_metrics(opts):
    loss_type = opts.get("loss_type")
    metrics_type = opts.get("metrics_type")

    loss = metrics = None

    if loss_type == "koopman":
        loss = KoopmanLoss(opts)
    elif loss_type == "spacetime":
        loss = SpaceTimeLoss(opts)
    else:
        raise ValueError("Unknown loss type: " + str(loss_type))

    if metrics_type == "spacetime":
        metrics = SpaceTimeMetrics(opts, True)
    else:
        raise ValueError("Unknown loss type: " + str(metrics_type))

    return loss, metrics
