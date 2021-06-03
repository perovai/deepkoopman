import numpy as np
import torch


class Encoder(torch.nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.hidden_layers = params["hidden_layers"]
        self.hidden_dim = params["hidden_dim"]
        self.input_dim = params["input_dim"]
        self.output_dim = params["latent_dim"]
        self.input_layer = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                for _ in range(self.hidden_layers - 1)
            ]
        )
        self.output_layer = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, inputs):
        x = torch.relu(self.input_layer(inputs))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        output = self.output_layer(x)

        return output


class Decoder(torch.nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.hidden_layers = params["hidden_layers"]
        self.hidden_dim = params["hidden_dim"]
        self.input_dim = 4
        self.output_dim = 4
        self.input_layer = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                for _ in range(self.hidden_layers - 1)
            ]
        )
        self.output_layer = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, inputs):
        x = torch.relu(self.input_layer(inputs))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        output = self.output_layer(x)

        return output


class Auxiliary(torch.nn.Module):
    def __init__(self, params):
        super(Auxiliary, self).__init__()
        self.hidden_layers = params["hidden_layers_aux"]
        self.hidden_dim = params["hidden_dim_aux"]
        self.input_dim = params["latent_dim"]
        self.output_dim = 10
        self.input_layer = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                for _ in range(self.hidden_layers - 1)
            ]
        )
        self.output_layer = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, inputs):
        x = torch.relu(self.input_layer(inputs))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        parameters = self.output_layer(x)
        koopman_operator = self.get_koopman(parameters)

        return koopman_operator

    def get_koopman(self, parameters):
        frequencies = parameters[:, :6]
        dampings = parameters[:, 6:]

        tri_indices = np.triu_indices(4, 1)
        diag_indices = np.diag_indices(4)
        koopman_log = torch.zeros(parameters.shape[0], 4, 4).to(parameters.device)
        koopman_damping = torch.zeros(parameters.shape[0], 4, 4).to(parameters.device)
        koopman_log[:, tri_indices[0], tri_indices[1]] = frequencies
        koopman_log -= koopman_log.permute(0, 2, 1)
        koopman_damping[:, diag_indices[0], diag_indices[1]] = dampings
        koopman_damping = torch.tanh(koopman_damping)

        koopman_rotation = torch.matrix_exp(koopman_log)
        koopman_operator = koopman_damping @ koopman_rotation

        return koopman_operator


class DeepKoopman(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.num_timesteps = params["sequence_length"]
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.auxiliary = Auxiliary(params)

    def forward(self, inputs):
        embedding = self.encoder(inputs)
        embedding_evolution = self.evolve_embedding(embedding)
        reconstruction = self.decoder(embedding)
        state_evolution = self.decoder(embedding_evolution)

        return reconstruction, embedding_evolution, state_evolution

    def evolve_embedding(self, embedding):
        embedding_evolution = torch.zeros(embedding.shape[0], self.num_timesteps, 4).to(
            embedding.device
        )
        for timestep in range(self.num_timesteps):
            koopman_operator = self.auxiliary(embedding)
            next_embedding = (koopman_operator @ embedding.unsqueeze(2)).squeeze()
            embedding_evolution[:, timestep, :] = next_embedding
            embedding = next_embedding

        return embedding_evolution

    def predict_next(self, inputs):
        embedding = self.encoder(inputs)
        koopman_operator = self.auxiliary(embedding)
        next_embedding = (koopman_operator @ embedding.unsqueeze(2)).squeeze()
        next_state = self.decoder(next_embedding)

        return next_state
