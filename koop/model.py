import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.hidden_layers = params["hidden_layers"]
        self.hidden_dim = params["hidden_dim"]
        self.input_dim = params["input_dim"]
        self.output_dim = params["latent_dim"]

        self.layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.hidden_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.hidden_layers = params["hidden_layers"]
        self.hidden_dim = params["hidden_dim"]
        self.input_dim = params["latent_dim"]
        self.output_dim = params["input_dim"]

        self.layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.hidden_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class Auxiliary(nn.Module):
    """
    This network takes the encoded state, outputs the parameters of the Koopman
    operator (omega and mu) and then builds the operator by matrix exponentiation
    (to produce a n-dimensional rotation matrix).
    It is called Lambda in the original paper.
    """

    def __init__(self, params):
        super(Auxiliary, self).__init__()
        self.hidden_layers = params["hidden_layers_aux"]
        self.hidden_dim = params["hidden_dim_aux"]
        self.input_dim = params["latent_dim"]

        self.tri_indices = np.triu_indices(self.input_dim, 1)
        self.diag_indices = np.diag_indices(self.input_dim)

        self.n_frequencies = len(self.tri_indices[0])
        self.n_dampings = len(self.diag_indices[0])

        self.output_dim = self.n_frequencies + self.n_dampings

        self.layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.hidden_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        koopman_params = self.layers(x)
        koopman_operator = self.get_koopman(koopman_params)

        return koopman_operator

    def get_koopman(self, koopman_params):

        frequencies = koopman_params[:, : self.n_frequencies]
        dampings = koopman_params[:, self.n_frequencies :]

        koopman_log = torch.zeros(
            koopman_params.shape[0],
            self.input_dim,
            self.input_dim,
            device=koopman_params.device,
        )
        koopman_damping = torch.zeros(
            koopman_params.shape[0],
            self.input_dim,
            self.input_dim,
            device=koopman_params.device,
        )
        koopman_log[:, self.tri_indices[0], self.tri_indices[1]] = frequencies
        koopman_log -= koopman_log.permute(0, 2, 1)
        koopman_damping[:, self.diag_indices[0], self.diag_indices[1]] = dampings
        koopman_damping = torch.tanh(koopman_damping)

        koopman_rotation = torch.matrix_exp(koopman_log)
        koopman_operator = koopman_damping @ koopman_rotation

        return koopman_operator


class DeepKoopman(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.n_shifts = params['num_shifts']
        self.shifts = np.arange(self.n_shifts) + 1
        self.num_timesteps = params["sequence_length"] - 1
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.auxiliary = Auxiliary(params)

    def forward(self, inputs):

        # https://github.com/BethanyL/DeepKoopman/blob/cad065179def289539d7ce162a261343895c9f99/networkarch.py#L412
        embedding = self.encoder(inputs) # latent state
        reconstruction = self.decoder(embedding) #

        koopman_operator = self.auxiliary.forward(embedding)
        next_embedding = (koopman_operator @ embedding.unsqueeze(2)).squeeze()
        breakpoint()

        state_evolution = []
        for j in np.arange(max(self.shifts)):
            if j + 1 in self.shifts:
                state_evolution.append(self.decoder(next_embedding))

            koopman_operator = self.auxiliary.forward(embedding)
            next_embedding = (koopman_operator @ embedding.unsqueeze(2)).squeeze()

        breakpoint()
        state_evolution = torch.cat([y.unsqueeze(1) for y in state_evolution], dim=1) # batch x n_shifts x input_dim
        return reconstruction, embedding_evolution, state_evolution

    def evolve_embedding(self, embedding):
        embedding_evolution = torch.zeros(
            embedding.shape[0],
            self.num_timesteps,
            self.auxiliary.input_dim,
            device=embedding.device,
        )
        for timestep in range(self.num_timesteps):
            koopman_operator = self.auxiliary.forward(embedding)
            next_embedding = (koopman_operator @ embedding.unsqueeze(2)).squeeze()
            embedding_evolution[:, timestep, :] = next_embedding
            embedding = next_embedding

        return embedding_evolution

    def predict_next(self, inputs):
        embedding = self.encoder(inputs)
        koopman_operator = self.auxiliary(embedding)
        next_embedding = (koopman_operator @ embedding.unsqueeze(2)).squeeze()
        next_state = self.decoder.forward(next_embedding)

        return next_state

    def infer_trajectory(
        self, initial_state, n_steps=None, return_intermediate_states=True
    ):
        if n_steps is None:
            n_steps = self.num_timesteps

        intermediate_states = []

        embedding = self.encoder.forward(initial_state)
        for _ in range(n_steps):
            koopman_operator = self.auxiliary.forward(embedding)
            next_embedding = (koopman_operator @ embedding.unsqueeze(2)).squeeze()
            if return_intermediate_states:
                next_state = self.decoder.forward(next_embedding)
                intermediate_states.append(next_state)
            embedding = next_embedding
        final_state = self.decoder.forward(embedding)

        if return_intermediate_states:
            return final_state, intermediate_states

        return final_state
