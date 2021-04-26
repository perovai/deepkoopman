import pathlib
import json

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

DATASET_PATH = pathlib.Path("./datasets")
LOSSES_PATH = pathlib.Path("./losses")
MODELS_PATH = pathlib.Path("./models")

# Set seed
torch.manual_seed(0)
np.random.seed(0)


def load_json(path):
    with open(path, "r+") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w+") as f:
        return json.dump(obj, f)


def convert_state(state):
    theta_1 = np.arctan2(state[0], state[1])
    theta_2 = np.arctan2(state[2], state[3])

    converted_state = [theta_1, theta_2] + state[-2:]

    return converted_state


def split_sequences(trajectories, sequence_length):
    sequences = []
    states = []
    for trajectory in trajectories:
        num_sequences = len(trajectory) - sequence_length
        for timestep in range(num_sequences):
            sequence = trajectory[timestep + 1 : timestep + sequence_length + 1]
            state = trajectory[timestep]

            sequence = [convert_state(state) for state in sequence]
            state = convert_state(state)

            sequences.append(sequence)
            states.append(state)

    return states, sequences


def load_datasets(path, sequence_length):
    dataset = load_json(path)
    train_trajectories = [
        [timestep["state"] for timestep in data] for data in dataset[:450]
    ]
    train_states, train_sequences = split_sequences(train_trajectories, sequence_length)
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_states), torch.FloatTensor(train_sequences)
    )

    valid_trajectories = [
        [timestep["state"] for timestep in data] for data in dataset[450:]
    ]
    valid_states, valid_sequences = split_sequences(valid_trajectories, sequence_length)
    validation_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(valid_states), torch.FloatTensor(valid_sequences)
    )
    return train_dataset, validation_dataset


def modified_linear(x):
    return torch.relu(x - 1) - torch.relu(-x - 1)


def get_reconstruction_loss(reconstruction, state):
    return torch.nn.functional.mse_loss(reconstruction, state)


def get_prediction_loss(state_evolution, sequence):
    return torch.nn.functional.mse_loss(state_evolution, sequence)


def get_koopman_loss(embedding_evolution, sequence, encoder):
    embedding_sequence = encoder(sequence)
    return torch.nn.functional.mse_loss(embedding_evolution, embedding_sequence)


class Encoder(torch.nn.Module):
    def __init__(self, hyperparams):
        super(Encoder, self).__init__()
        self.hidden_layers = hyperparams["hidden_layers"]
        self.hidden_dim = hyperparams["hidden_dim"]
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


class Decoder(torch.nn.Module):
    def __init__(self, hyperparams):
        super(Decoder, self).__init__()
        self.hidden_layers = hyperparams["hidden_layers"]
        self.hidden_dim = hyperparams["hidden_dim"]
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


class Auxiliairy(torch.nn.Module):
    def __init__(self, hyperparams):
        super(Auxiliairy, self).__init__()
        self.hidden_layers = hyperparams["hidden_layers_aux"]
        self.hidden_dim = hyperparams["hidden_dim_aux"]
        self.input_dim = 4
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
        frequenties = parameters[:, :6]
        dampings = parameters[:, 6:]

        tri_indices = np.triu_indices(4, 1)
        diag_indices = np.diag_indices(4)
        koopman_log = torch.zeros(parameters.shape[0], 4, 4).to(parameters.device)
        koopman_damping = torch.zeros(parameters.shape[0], 4, 4).to(parameters.device)
        koopman_log[:, tri_indices[0], tri_indices[1]] = frequenties
        koopman_log -= koopman_log.permute(0, 2, 1)
        koopman_damping[:, diag_indices[0], diag_indices[1]] = dampings
        koopman_damping = torch.tanh(koopman_damping)

        koopman_rotation = torch.matrix_exp(koopman_log)
        koopman_operator = koopman_damping @ koopman_rotation

        return koopman_operator


class DeepKoopman(torch.nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.num_timesteps = hyperparams["sequence_length"]
        self.encoder = Encoder(hyperparams)
        self.decoder = Decoder(hyperparams)
        self.auxiliairy = Auxiliairy(hyperparams)

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
            koopman_operator = self.auxiliairy(embedding)
            next_embedding = (koopman_operator @ embedding.unsqueeze(2)).squeeze()
            embedding_evolution[:, timestep, :] = next_embedding
            embedding = next_embedding

        return embedding_evolution

    def predict_next(self, inputs):
        embedding = self.encoder(inputs)
        koopman_operator = self.auxiliairy(embedding)
        next_embedding = (koopman_operator @ embedding.unsqueeze(2)).squeeze()
        next_state = self.decoder(next_embedding)

        return next_state


def train_and_monitor(hyperparams):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("GPU requested but not available")

    print(f"Loading dataset {hyperparams['dataset']}_policy_trajectories.json")
    train_dataset, validation_dataset = load_datasets(
        DATASET_PATH / f"{hyperparams['dataset']}_policy_trajectories.json",
        hyperparams["sequence_length"],
    )
    print("Dataset loaded")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=False,
    )

    model = DeepKoopman(hyperparams)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=20,
        min_lr=1e-6,
    )

    metrics = {}
    training_losses = []
    reconstruction_losses = []
    prediction_losses = []
    koopman_losses = []
    validation_losses = []

    progress = tqdm(
        range(1, hyperparams["epochs"] + 1),
        desc="Loss: ",
        total=len(train_loader),
        position=0,
        leave=True,
    )

    for epoch in range(1, hyperparams["epochs"] + 1):
        print(f"epoch: {epoch}")

        model.train()

        total_loss = 0
        total_reconstruction_loss = 0
        total_prediction_loss = 0
        total_koopman_loss = 0

        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            position=0,
            leave=True,
        )

        for i, ground_truth in progress:
            state, sequence = ground_truth
            state = state.to(device)
            sequence = sequence.to(device)
            optimizer.zero_grad()

            reconstruction, embedding_evolution, state_evolution = model(state)

            reconstruction_loss = get_reconstruction_loss(reconstruction, state)
            prediction_loss = get_prediction_loss(state_evolution, sequence)
            # diff = state_evolution - sequence
            # max_diff = torch.max(diff)
            koopman_loss = get_koopman_loss(
                embedding_evolution,
                sequence,
                model.encoder,
            )
            loss = (
                hyperparams["reconstruction_weight"] * reconstruction_loss
                + hyperparams["prediction_weight"] * prediction_loss
                + hyperparams["koopman_weight"] * koopman_loss
                if epoch > hyperparams["reconstruction_epochs"]
                else reconstruction_loss
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_prediction_loss += prediction_loss.item()
            total_koopman_loss += koopman_loss.item()
            # progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))
        mean_train_loss = total_loss / len(train_loader)
        mean_reconstruction_loss = total_reconstruction_loss / len(train_loader)
        mean_prediction_loss = total_prediction_loss / len(train_loader)
        mean_koopman_loss = total_koopman_loss / len(train_loader)
        training_losses.append(mean_train_loss)
        reconstruction_losses.append(mean_reconstruction_loss)
        prediction_losses.append(mean_prediction_loss)
        koopman_losses.append(mean_koopman_loss)
        if epoch > hyperparams["reconstruction_epochs"]:
            print(f"    reconstruction loss: {mean_reconstruction_loss:.5f}")
            print(f"    prediction loss: {mean_prediction_loss:.5f}")
            print(f"    koopman loss: {mean_koopman_loss:.5f}")
        print(f"    train loss: {mean_train_loss:.5f}")

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for ground_truth in validation_loader:
                state, sequence = ground_truth
                state = state.to(device)
                sequence = sequence.to(device)
                optimizer.zero_grad()

                reconstruction, embedding_evolution, state_evolution = model(state)

                reconstruction_loss = get_reconstruction_loss(reconstruction, state)
                prediction_loss = get_prediction_loss(state_evolution, sequence)
                koopman_loss = get_koopman_loss(
                    embedding_evolution,
                    sequence,
                    model.encoder,
                )
                loss = (
                    hyperparams["reconstruction_weight"] * reconstruction_loss
                    + hyperparams["prediction_weight"] * prediction_loss
                    + hyperparams["koopman_weight"] * koopman_loss
                    if epoch > hyperparams["reconstruction_epochs"]
                    else reconstruction_loss
                )

                total_loss += loss.item()
            save_json(
                metrics,
                LOSSES_PATH / f"koopman_training_metrics_{hyperparams['dataset']}.json",
            )
        mean_valid_loss = total_loss / len(validation_loader)
        scheduler.step(mean_valid_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"    learning rate: {learning_rate:.5f}")
        validation_losses.append(mean_valid_loss)
        print(f"    valid loss: {mean_valid_loss:.5f}")
        if (
            epoch > hyperparams["reconstruction_epochs"]
            and validation_losses[-1] < validation_losses[-2]
        ):
            torch.save(
                model,
                MODELS_PATH
                / f"best_koopman_prediction_model_{hyperparams['dataset']}.pt",
            )
    metrics["Train loss"] = training_losses
    metrics["Reconstruction loss"] = reconstruction_losses
    metrics["Prediction loss"] = prediction_losses
    metrics["Koopman loss"] = koopman_losses
    metrics["Validation loss"] = validation_losses
    save_json(
        metrics, LOSSES_PATH / f"koopman_training_metrics_{hyperparams['dataset']}.json"
    )

    print("\nOptimization ended.\n")


def plot_losses(losses_path):
    metrics = load_json(losses_path)
    training_losses = metrics["Train loss"]

    epochs = np.arange(len(training_losses))
    for loss_name, losses in metrics.items():
        plt.plot(epochs, losses, label=loss_name)
    plt.ylim([0, 3])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_trajectory(dataset_path, model_path):
    dataset = load_json(dataset_path)
    model = torch.load(model_path, map_location=torch.device("cpu"))
    trajectory = [timestep["state"] for timestep in dataset[456]][:50]
    trajectory = np.array([convert_state(state) for state in trajectory])
    trajectory_tensor = torch.FloatTensor(trajectory)

    state = trajectory_tensor[0, :]
    initial_state = state
    predicticted_trajectory = torch.zeros_like(trajectory_tensor)

    for timestep in range(len(trajectory)):
        predicticted_trajectory[timestep, :] = state
        next_state = model.predict_next(state.unsqueeze(0))
        state = next_state

    predicticted_trajectory = predicticted_trajectory.detach().numpy()

    theta_1 = trajectory.T[0]
    theta_2 = trajectory.T[1]
    predicted_theta_1 = predicticted_trajectory.T[0]
    predicted_theta_2 = predicticted_trajectory.T[1]

    # plt.plot(theta_1, theta_2, label="True trajectory", linewidth=0.2)
    plt.quiver(
        theta_1[:-1],
        theta_2[:-1],
        theta_1[1:] - theta_1[:-1],
        theta_2[1:] - theta_2[:-1],
        scale_units="xy",
        angles="xy",
        scale=1,
        width=0.0025,
        headwidth=5,
        color="b",
        label="Predicted trajectory",
    )
    # plt.plot(
    #     predicted_theta_1,
    #     predicted_theta_2,
    #     label="Predicted trajectory",
    #     linewidth=0.2,
    # )
    plt.quiver(
        predicted_theta_1[:-1],
        predicted_theta_2[:-1],
        predicted_theta_1[1:] - predicted_theta_1[:-1],
        predicted_theta_2[1:] - predicted_theta_2[:-1],
        scale_units="xy",
        angles="xy",
        scale=1,
        width=0.0025,
        headwidth=5,
        color="orange",
        label="Predicted trajectory",
    )
    plt.scatter(theta_1, theta_2, label="True trajectory", s=5, c="b")
    plt.scatter(
        predicted_theta_1,
        predicted_theta_2,
        label="Predicted trajectory",
        s=5,
        c="orange",
    )
    plt.scatter(initial_state[0], initial_state[1], s=50, c="g")
    plt.ylim([0, np.pi])
    plt.xlim([0, np.pi])
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.show()


def main():
    hyperparams = {
        "dataset": "deterministic",
        "sequence_length": 32,
        "batch_size": 256,
        "hidden_layers": 3,
        "hidden_dim": 100,
        "hidden_layers_aux": 3,
        "hidden_dim_aux": 50,
        "lr": 2e-3,
        "reconstruction_weight": 1,
        "prediction_weight": 1,
        "koopman_weight": 1,
        "epochs": 100,
        "reconstruction_epochs": 5,
    }

    train_and_monitor(hyperparams)

    # plot_losses(LOSSES_PATH / "koopman_training_metrics_deterministic.json")
    # plot_trajectory(
    #     DATASET_PATH / "deterministic_policy_trajectories.json",
    #     MODELS_PATH / "best_koopman_prediction_model_deterministic.pt",
    # )


if __name__ == "__main__":
    main()
