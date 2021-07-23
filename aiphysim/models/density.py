import torch.nn as nn

from aiphysim.utils import timeit


class Encoder(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.input_dim = opts.input_dim
        self.output_dim = opts.latent_dim
        widths = opts.encoder_widths

        self.layers = [nn.Linear(self.input_dim, widths[0]), nn.ReLU()]
        for j in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[j], widths[j + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(widths[j + 1], self.output_dim))
        self.layers = nn.Sequential(*self.layers)

    @timeit
    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.input_dim = opts.latent_dim
        self.output_dim = opts.input_dim
        widths = opts.encoder_widths[::-1]

        self.layers = [nn.Linear(self.input_dim, widths[0]), nn.ReLU()]
        for j in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[j], widths[j + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(widths[j + 1], self.output_dim))
        self.layers = nn.Sequential(*self.layers)

    @timeit
    def forward(self, x):
        return self.layers(x)


class Latent(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        input_size = opts.latent_dim
        hidden_size = opts.latent_hidden_size
        num_layers = opts.latent_num_layers
        dropout = opts.dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            proj_size=input_size,
        )

    @timeit
    def forward(self, z):
        if z.ndim == 2:
            z.unsqueeze_(1)
        return self.lstm(z)


class DynamicLatentModel(nn.Module):
    def __init__(self, opts):
        super().__init__()

        self.opts = opts

        self.encoder = Encoder(opts)
        self.decoder = Decoder(opts)
        self.latent = Latent(opts)
        self.density_matrix_shape = opts.density_matrix_shape

        self.flatten = nn.Flatten(-2, -1)
        self.unflatten = nn.Unflatten(-1, tuple(self.density_matrix_shape))

    @timeit
    def encode(self, x):
        x = self.flatten(x)
        z = self.encoder(x)
        return z

    @timeit
    def decode(self, z):
        y = self.decoder(z)
        return self.unflatten(y)

    def next_z(self, z):
        last, _ = self.latent(z)
        return last

    def evolve_from_state(self, state, steps=0):
        latents = [self.encode(state)]
        decoded = [self.decode(latents[-1])]
        for _ in range(steps):
            z = latents[-1]
            latents.append(self.next_z(z))
            decoded.append(self.decode(z))

        return latents, decoded

    def one_step_predictions(self, ts):
        encoded_ts = self.encode(ts)
        decoded_ts = self.decode(encoded_ts)
        next_zs, _ = self.latent(encoded_ts)
        next_decoded_ts = self.decode(next_zs)
        return encoded_ts, decoded_ts, next_zs, next_decoded_ts

    @timeit
    def forward(self, batch, device="cpu"):
        time_series = batch["data"].to(device)
        return self.one_step_predictions(time_series)
