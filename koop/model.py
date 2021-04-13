import torch.nn as nn

from koop.utils import init_weights


class Network(nn.Sequential):
    def __init__(self, widths=[], dist_weights=[], dist_biases=[], scale=1):
        super().__init__()

        self.dist_weights = dist_weights
        self.dist_biases = dist_biases
        self.scale = scale

        for i, (in_d, out_d) in enumerate(zip(widths[:-1], widths[1:])):
            self.add_module(f"Layer-{i}", nn.Linear(in_d, out_d))
            if i < len(widths) - 2:
                self.add_module(f"ReLU-{i}", nn.ReLU())

        self.init_parameters()

    def init_parameters(self):
        for k, module in enumerate(self.children()):
            wd = (
                self.dist_weights[k]
                if isinstance(self.dist_weights[k], list)
                else self.dist_weights
            )
            bd = (
                self.dist_biases[k]
                if isinstance(self.dist_biases[k], list)
                else self.dist_biases
            )
            init_weights(module, wd, bd, self.scale)


class AutoEncoder(nn.Module):
    def __init__(self, widths, dist_weights, dist_biases, scale):
        super().__init__()

        self.encoder = Network(widths, dist_weights, dist_biases, scale)
        self.decoder = Network(
            widths[::-1],
            dist_weights[::-1] if isinstance(dist_weights, list) else dist_weights,
            dist_biases[::-1] if isinstance(dist_biases, list) else dist_biases,
            scale,
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y
