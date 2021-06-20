import numpy as np
import torch
import torch.nn as nn

def get_latent_dim(params):
    return 2 * params['num_complex_pairs'] + params['num_real']


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        # self.hidden_layers = params["hidden_layers"]
        # self.hidden_dim = params["hidden_dim"]
        self.input_dim = params["input_dim"]
        # self.output_dim = params["latent_dim"]
        widths = params["encoder_widths"]

        self.output_dim = latent_dim = get_latent_dim(params)

        self.layers = [nn.Linear(self.input_dim, widths[0]), nn.ReLU()]
        for j in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[j], widths[j+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(widths[j+1], latent_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        # self.hidden_layers = params["hidden_layers"]
        # self.hidden_dim = params["hidden_dim"]
        # self.input_dim = params["latent_dim"]
        # self.output_dim = params["input_dim"]

        latent_dim = self.input_dim = get_latent_dim(params)
        self.output_dim = params["input_dim"]
        widths = params["decoder_widths"]

        self.layers = [nn.Linear(latent_dim, widths[0]), nn.ReLU()]
        for j in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[j], widths[j+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(widths[j+1], self.output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

class BasisOmegaNet(nn.Module):
    """
    Creates an omega net for the `type` of eigenfunctions

    Args:
        width_omega (list): widths of hidden layers
        type (str): "complex" for complex eigenfunctions, "real" for real eigenfunctions

    Returns:
        torch.nn.Sequential
    """
    def __init__(self, widths_omega, omega_type):
        super(BasisOmegaNet, self).__init__()

        if omega_type == "complex":
            output_dim = 2
        elif omega_type == "real":
            output_dim = 1
        else:
            raise ValueError(f"Unexpected omega net type: {omega_type}")

        layers = [nn.Linear(1, widths_omega[0]), nn.ReLU()]

        for j in range(len(widths_omega)-1):
            layers.append(nn.Linear(widths_omega[j], widths_omega[j+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(widths_omega[j+1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class OmegaNet(nn.Module):
    def __init__(self, params):
        super(OmegaNet, self).__init__()
        self.n_complex_pairs = params.num_complex_pairs
        self.n_real = params.num_real
        self.delta_t = params.delta_t
        self.latent_dim = get_latent_dim(params)

        self.complex_net = {}
        for j in range(self.n_complex_pairs):
            self.complex_net[j+1] = BasisOmegaNet(params.widths_omega_complex, omega_type="complex") # self.create_omega_net(params.widths_omega_complex, omega_type="complex")
            super(OmegaNet, self).add_module(name=f"OC_{j+1}", module=self.complex_net[j+1])

        self.real_net = {}
        for j in range(self.n_real):
            self.real_net[j+1] = BasisOmegaNet(params.widths_omega_real, omega_type="real")
            super(OmegaNet, self).add_module(name=f"OR_{j+1}", module=self.real_net[j+1])

    def get_omegas(self, y):
        """
        Creates Koopman eigenfunctions based on the latent coordinates

        Args:
            y_t (torch.tensor): (seq. x batch x input_dim) encoded representation of the inputs

        Returns:
            y_{t+1} (torch.tensor): (seq. x batch x input_dim) advanced state of each encoded input
        """
        omegas = []

        # pair of coords --> complex kopman eigenfunctions
        for j in np.arange(self.n_complex_pairs):
            ind = 2 * j
            pair_of_columns = y[:, ind:ind+2]
            radius_of_pair = torch.sum(torch.square(pair_of_columns), axis=1, keep_dims=True)
            omegas.append(self.complex_net[j+1](radius_of_pair))

        # coords --> real kopman eigenfunctions
        for j in np.arange(self.n_real):
            ind = 2 * self.n_complex_pairs + j
            omegas.append(self.real_net[j+1](y[:, ind:ind+1]))

        return omegas

    def forward(self, y):
        """
        Advances the latent coordinates based on the omega at those coordinates

        Args:
            y_t (torch.tensor): (None, input_dim) encoded representation of the inputs

        Returns:
            y_{t+1} (torch.tensor): (batch x input_dim) advanced state of each encoded input
        """
        omegas = self.get_omegas(y)

        # Advance coordinates in the space of complex eigenvectors
        complex_list = []
        for j in np.arange(self.n_complex_pairs):
            ind = 2 * j
            y_stack = torch.stack([y[:, ind:ind+2], y[:, ind: ind+2]], axis=2)
            L_stack = self.form_complex_conjugate_block(omegas[j], self.delta_t)
            complex_list.append(torch.sum(y_stack * L_stack, axis=1))

        # coords --> real kopman eigenfunctions
        real_list = []
        for j in np.arange(self.n_real):
            ind = 2 * self.n_complex_pairs + j
            y_real_next = y[:, ind:ind+1] * torch.exp(omegas[self.n_complex_pairs + j] * self.delta_t)
            real_list.append(y_real_next)

        y_next = torch.Tensor().to(y.device)
        if complex_list:
            y_next = torch.cat([y_next, torch.cat(complex_list, dim=1)], dim=1)

        if real_list:
            y_next = torch.cat([y_next, torch.cat(real_list, dim=1)], dim=1)

        return y_next.reshape(-1, self.latent_dim)

    @staticmethod
    def form_complex_conjugate_block(self, omegas, delta_t):
        """
        Forms the block of rotation matrix to advance the state in the complex eigenfunctions
        """
        scale = torch.exp(omegas[:, 1] * delta_t) # [None, 1]
        a11 = scale * torch.cos(omegas[:, 0] * delta_t) # [None, 1]
        a12 = scale * torch.sin(omegas[:,0] * delta_t) # [None, 1]
        row1 = torch.stack([a11, -a12], axis=1) # [None, 2]
        row2 = torch.stack([a12, a11], axis=1) # [None, 2]
        return torch.stack([row1, row2], axis=2) # [None, 2, 2]

class DeepKoopman(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.n_shifts = params['num_shifts']
        self.shifts = np.arange(self.n_shifts) + 1

        self.n_middle_shifts = params["num_shifts_middle"]
        self.middle_shifts = np.arange(self.n_middle_shifts) + 1

        self.max_shifts = max(max(self.middle_shifts), max(self.shifts))

        self.num_timesteps = params["sequence_length"] - 1
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.omega_net = OmegaNet(params)

        self.latent_dim = get_latent_dim(params)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.tensor): (seq x batch x input_dim)

        Returns:
            embedding (list): size: seq, where each element is a tensor of size (batch x latent_dim)
            y (list): size: `self.n_shifts`, predicted coordinates after some time steps determined by `self.shifts`
        """
        # https://github.com/BethanyL/DeepKoopman/blob/cad065179def289539d7ce162a261343895c9f99/networkarch.py#L412
        embedding = self.encoder(inputs) # latent state

        y = []
        latent_space_evol = []

        # y[0] is x[0,:,:] encoded and then decoded (no stepping forward); shape is batch x input_dim
        y.append(self.decoder(embedding[0, :, :]))

        #
        next_embedding = self.omega_net(embedding[0, :, :])

        for j in np.arange(self.max_shifts):
            if j + 1 in self.middle_shifts:
                latent_space_evol.append(next_embedding)

            if j + 1 in self.shifts:
                y.append(self.decoder(next_embedding))

            next_embedding = self.omega_net(next_embedding)

        return embedding, y, latent_space_evol

    def infer_trajectory(
        self, initial_state, n_steps=None, return_intermediate_states=True
    ):
        if n_steps is None:
            n_steps = self.num_timesteps

        intermediate_states = []

        embedding = self.encoder(initial_state)
        for _ in range(n_steps):
            next_embedding = self.omega_net(embedding)

            if return_intermediate_states:
                next_state = self.decoder(next_embedding)
                intermediate_states.append(next_state)

            embedding = next_embedding

        final_state = self.decoder(embedding)

        if return_intermediate_states:
            return final_state, intermediate_states

        return final_state
