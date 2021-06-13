import torch
import numpy as np

def get_reconstruction_loss(reconstruction, state):
    return torch.nn.functional.mse_loss(reconstruction, state)


def get_prediction_loss(state_evolution, sequence):
    return torch.nn.functional.mse_loss(state_evolution, sequence)


def get_koopman_loss(embedding_evolution, sequence, encoder):
    embedding_sequence = encoder(sequence)
    return torch.nn.functional.mse_loss(embedding_evolution, embedding_sequence)


class Loss:
    def __init__(self, opts):
        self.params = opts
        self.shifts = np.arange(self.params['num_shifts']) + 1
        self.weights = opts.weights
        self.reconstruction = (
            get_reconstruction_loss if opts.losses.get("reconstruction") else None
        )
        self.koopman = get_koopman_loss if opts.losses.get("koopman") else None
        self.prediction = get_prediction_loss if opts.losses.get("prediction") else None

    def compute(self, targets, predictions, model=None):
        """
        Compute the weighted loss with respect to targets and predictions

        Args:
            targets (dict): dictionnary of target values
            predictions (dict): dictionnary of predicted values
        """
        reconstruction, embedding_evolution, state_evolution = predictions
        state = targets[:, 0, ...]
        sequence = targets[:, 1:, ...]

        losses = {"total": 0.0}

        if self.weights.get("reconstruction", 0) > 0:
            losses["reconstruction"] = self.reconstruction(reconstruction, state)
            losses["total"] += self.weights.get("reconstruction") * losses["reconstruction"]

        # https://github.com/BethanyL/DeepKoopman/blob/master/training.py#L48
        # Weird : this loss is only with respect to initial condition. It should have been a rolling window
        if self.weights.get("prediction", 0) > 0:
            for j in range(self.params['num_shifts']):
                shift = self.shifts[j]
                losses["prediction"] += torch.nn.functional.mse_loss(state_evolution[j], targets[:, shift, ...])

            losses["prediction"] /= self.params["n_shifts"]
            losses["total"] += self.weights.get("prediction") * losses["prediction"]

        # https://github.com/BethanyL/DeepKoopman/blob/master/training.py#L62
        if self.weights.get("koopman", 0) > 0:
            assert isinstance(model, torch.nn.Module)
            assert hasattr(model, "encoder")
            # WIP
            # for j in np.arange(max(self.params['shifts_middle'])):
            #     if (j + 1) in params['shifts_middle']:
            #         losses["koopman"] +=
            losses["koopman"] = self.koopman(
                embedding_evolution,
                sequence,
                model.encoder,
            )
            losses["total"] += self.weights.get("koopman") * losses["koopman"]

        if self.weights.get("l2", 0) > 0:
            assert model is not None, "model is needed to regularize them"

            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param)

            losses['l2_norm'] = l2_reg
            losses['total'] += self.weights.get("l2") * l2_reg

        # inf norm on autoencoder error and one prediction step
        if self.weights.get("inf_norm", 0) > 0:
            inf = float("inf")
            Linf1_penalty = torch.norm(torch.norm(state_evolution[:, 0] - state, p=inf), p=inf)
            Linf2_penalty = torch.norm(torch.norm(state_evolution[:, 1] - sequence, p=inf), p=inf)
            losses['inf_loss'] = (Linf1_penalty + Linf2_penalty)
            losses['total'] += self.weights.get("inf_norm") * losses['inf_loss']

        return losses
