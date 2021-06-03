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
        if opts.losses.get("reconstruction"):
            self.reconstruction = get_reconstruction_loss
        if opts.losses.get("prediction"):
            self.prediction = get_prediction_loss
        if opts.losses.get("koopman"):
            self.koopman = get_koopman_loss
