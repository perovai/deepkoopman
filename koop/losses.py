import torch
import numpy as np

class Loss:
    def __init__(self, opts):
        self.params = opts

        self.n_shifts = opts['num_shifts']
        self.shifts = np.arange(self.n_shifts) + 1

        self.n_middle_shifts = opts["num_shifts_middle"]
        self.middle_shifts = np.arange(self.n_middle_shifts) + 1

        self.weights = opts.weights

    def compute(self, inputs, predictions, model=None):
        """
        Compute the weighted loss with respect to targets and predictions

        Args:
            targets (dict): dictionnary of target values
            predictions (dict): dictionnary of predicted values
        """
        embedding, y, latent_evol = predictions

        losses = {"total": 0.0}

        # autoencoder loss
        if self.weights.get("reconstruction", 0) > 0:
            losses["reconstruction"] = torch.nn.functional.mse_loss(inputs[0, :, :], y[0])
            losses["total"] += self.weights.get("reconstruction") * losses["reconstruction"]

        # https://github.com/BethanyL/DeepKoopman/blob/master/training.py#L48
        # dynamics / prediction loss
        if self.weights.get("prediction", 0) > 0:
            losses["prediction"] = 0
            for shift in self.shifts:
                losses["prediction"] += torch.nn.functional.mse_loss(inputs[shift, :, :], y[shift])
            losses["prediction"] /= self.n_shifts
            losses["total"] += self.weights.get("prediction") * losses["prediction"]

        # https://github.com/BethanyL/DeepKoopman/blob/master/training.py#L62
        # linear loss
        if self.weights.get("linear", 0) > 0:
            losses["linear"] = 0
            for j in np.arange(self.n_middle_shifts):
                shift = self.middle_shifts[j]
                losses["linear"] += torch.nn.functional.mse_loss(embedding[shift, :, :], latent_evol[j])

            losses["linear"] /= self.n_middle_shifts
            losses["total"] += self.weights.get("linear") * losses["linear"]

        # TODO: config should take in scientific notation
        l2_weights = eval(self.weights.get("l2", 0))
        if l2_weights > 0:
            assert model is not None, "model is needed to regularize them"

            l2_reg = torch.tensor(0.).to(inputs.device)
            for param in model.parameters():
                l2_reg += torch.norm(param)

            losses['l2_norm'] = l2_reg
            losses['total'] += l2_weights * l2_reg

        # inf norm on autoencoder error and one prediction step
        # TODO: same as above
        inf_weight = eval(self.weights.get("inf_norm", 0))
        if inf_weight > 0:
            inf = float("inf")
            Linf1_penalty = torch.norm(torch.norm(y[0] - inputs[0, :, :], p=inf), p=inf)
            Linf2_penalty = torch.norm(torch.norm(y[1] - inputs[1, :, :], p=inf), p=inf)
            losses['inf_loss'] = (Linf1_penalty + Linf2_penalty)
            losses['total'] += inf_weight * losses['inf_loss']

        return losses
