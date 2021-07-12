import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, opts):
        super(Encoder, self).__init__()
        self.input_dim = opts.input_dim
        self.output_dim = opts.latent_dim
        widths = opts.encoder_widths

        self.layers = [nn.Linear(self.input_dim, widths[0]), nn.ReLU()]
        for j in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[j], widths[j + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(widths[j + 1], self.output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, opts):
        super(Encoder, self).__init__()
        self.input_dim = opts.latent_dim
        self.output_dim = opts.input_dim
        widths = opts.encoder_widths[::-1]

        self.layers = [nn.Linear(self.input_dim, widths[0]), nn.ReLU()]
        for j in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[j], widths[j + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(widths[j + 1], self.output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class Latent(nn.Module):
    def __init__(self, opts):
        self.opts = opts

    # TODO


class DynamicLatentModel(nn.Module):
    def __init__(self, opts):
        self.opts = opts

        self.encoder = Encoder(opts)
        self.decoder = Decoder(opts)
        self.latent = Latent(opts)
        self.density_matrix_shape = opts.density_matrix_shape

        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, tuple(self.density_matrix_shape))

    def encode(self, x):
        z = self.encoder(x)
        return self.flatten(z)

    def decode(self, z):
        y = self.decoder(z)
        return self.unflatten(y)

    def next_z(self, z):
        return z
        # return self.latent(z)

    def evolve(self, x, steps=0):
        latents = [self.encode(x)]
        decoded = [self.decode(latents[-1])]
        for s in range(steps):
            latents.append(self.next_z(z))
            decoded.append(self.decode(latents[-1]))

        return latents, decoded
