import traceback as tb

import numpy as np
import torch


class BaseLoss:
    def __init__(self, opts):
        self.weights = {name: float(weight) for name, weight in opts.weights.items()}
        self.loss_funcs = {
            loss_name: getattr(BaseLoss, loss_name) for loss_name in opts.losses
        }
        self.shifts = np.arange(opts.num_shifts) + 1

    @staticmethod
    def reconstruction(reconstruction, state):
        return torch.nn.functional.mse_loss(reconstruction, state)

    @staticmethod
    def prediction(state_evolution, targets, shifts):
        pred_loss = torch.tensor(0.0)
        for j, shift in enumerate(shifts):
            pred_loss += torch.nn.functional.mse_loss(
                state_evolution[:, j, ...], targets[:, shift, ...]
            )

        pred_loss = pred_loss / len(shifts)
        return pred_loss

    @staticmethod
    def latent_evolution(embedding_evolution, sequence, encoder):
        embedding_sequence = encoder(sequence)
        return torch.nn.functional.mse_loss(embedding_evolution, embedding_sequence)

    @staticmethod
    def inf_norm(state_evolution, state, sequence):
        inf = float("inf")
        inf1_penalty = torch.norm(
            torch.norm(state_evolution[:, 0] - state, p=inf), p=inf
        )
        return inf1_penalty
        # inf2_penalty = torch.norm(
        #     torch.norm(state_evolution[:, 1] - sequence, p=inf), p=inf
        # )  # @Prateek: shape mismatch: batch x dim vs batch x seq x dim
        # return inf1_penalty + inf2_penalty

    @staticmethod
    def l2(model):
        l2_reg = torch.tensor(0.0)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        return l2_reg

    def compute(self, *args):
        self.set_args(*args)

        losses = {"total": 0.0}

        for name, loss in self.loss_funcs.items():
            if name in self.args:
                try:
                    losses[name] = loss(*self.args[name])
                    losses["total"] += self.weights.get(name, 1) * losses[name]
                except Exception as e:
                    print(
                        "Error in loss", name, "with weight", self.weights.get(name, 1)
                    )
                    print("\n" + tb.format_exc())
                    raise Exception(e)
        return losses


class KoopmanLoss(BaseLoss):
    def set_args(self, targets, predictions, model):
        """
        Compute the weighted loss with respect to targets and predictions

        Args:
            targets (dict): dictionnary of target values
            predictions (dict): dictionnary of predicted values
        """
        reconstruction, embedding_evolution, state_evolution = predictions
        state = targets[:, 0, ...]
        sequence = targets[:, 1:, ...]

        self.args = {
            "reconstruction": (reconstruction, state),
            "prediction": (state_evolution, targets, self.shifts),
            "latent_evolution": (
                embedding_evolution,
                sequence,
                model.encoder,
            ),
            "l2": (model,),
            "inf_norm": (state_evolution, state, sequence),
        }


def get_loss(opts):
    loss_type = opts.get("loss_type")
    if loss_type == "koopman":
        return KoopmanLoss(opts)

    raise ValueError("Unknown loss type: " + str(loss_type))
