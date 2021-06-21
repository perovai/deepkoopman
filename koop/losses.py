import torch


def get_reconstruction_loss(reconstruction, state):
    return torch.nn.functional.mse_loss(reconstruction, state)


def get_prediction_loss(state_evolution, sequence):
    return torch.nn.functional.mse_loss(state_evolution, sequence)


def get_koopman_loss(embedding_evolution, sequence, encoder):
    embedding_sequence = encoder(sequence)
    return torch.nn.functional.mse_loss(embedding_evolution, embedding_sequence)


class Loss:
    def __init__(self, opts):
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

        if self.reconstruction:
            losses["reconstruction"] = self.reconstruction(reconstruction, state)
            losses["total"] += (
                self.weights.get("reconstruction", 1) * losses["reconstruction"]
            )

        if self.prediction:
            losses["prediction"] = self.prediction(state_evolution, sequence)
            losses["total"] += self.weights.get("prediction", 1) * losses["prediction"]

        if self.koopman:
            assert isinstance(model, torch.nn.Module)
            assert hasattr(model, "encoder")

            losses["koopman"] = self.koopman(
                embedding_evolution,
                sequence,
                model.encoder,
            )
            losses["total"] += self.weights.get("koopman", 1) * losses["koopman"]

        return losses
