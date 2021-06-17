import torch


class BaseLoss:
    def __init__(self, opts):
        self.weights = opts.weights
        self.loss_funcs = {
            loss_name: getattr(BaseLoss, loss_name)
            for loss_name, should_use in opts.losses.items()
            if should_use
        }

    @staticmethod
    def reconstruction(reconstruction, state):
        return torch.nn.functional.mse_loss(reconstruction, state)

    @staticmethod
    def prediction(state_evolution, sequence):
        return torch.nn.functional.mse_loss(state_evolution, sequence)

    @staticmethod
    def latent_evolution(embedding_evolution, sequence, encoder):
        embedding_sequence = encoder(sequence)
        return torch.nn.functional.mse_loss(embedding_evolution, embedding_sequence)

    def compute(self, *args):
        self.set_args(*args)

        losses = {"total": 0.0}

        for name, loss in self.loss_funcs.items():
            losses[name] = loss(*self.args[name])
            losses["total"] += self.weights.get(name, 1) * losses[name]

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
            "prediction": (state_evolution, sequence),
            "latent_evolution": (
                embedding_evolution,
                sequence,
                model.encoder,
            ),
        }


def get_loss(opts):
    loss_type = opts.get("loss_type")
    if loss_type == "koopman":
        return KoopmanLoss(opts)

    raise ValueError("Unknown loss type: " + str(loss_type))
