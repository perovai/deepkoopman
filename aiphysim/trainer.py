import numpy as np
import torch
from comet_ml import ExistingExperiment, Experiment
from tqdm import tqdm

from aiphysim.dataloading import create_dataloaders
from aiphysim.plots import plot_2D_comparative_trajectories
from aiphysim.logger import Logger
from aiphysim.losses import get_loss
from aiphysim.models import create_model
from aiphysim.opts import Opts
from aiphysim.utils import (
    COMET_KWARGS,
    clean_checkpoints,
    find_existing_comet_id,
    get_optimizer,
    load_opts,
    mem_size,
    num_params,
    resolve,
)


class Trainer:
    def __init__(self, opts, exp=None) -> None:
        self.opts = opts
        self.is_setup = False
        self.exp = exp

        self.early_stop = False
        self.early_score = None
        self.early_counter = 0

    @classmethod
    def resume_from_path(
        cls, path, overrides=None, exp_type="resume", inference_only=False
    ):
        path = resolve(path)

        ckpt = "latest.ckpt"
        if path.is_file():
            assert path.suffix == ".ckpt"
            ckpt = path.name
            path = path.parent

        opts_path = path / "opts.yaml"
        assert opts_path.exists()
        opts = load_opts(opts_path)
        breakpoint()
        if isinstance(overrides, (Opts, dict)):
            opts.update(overrides)

        if exp_type == "new":
            exp = Experiment(
                workspace=opts.comet.workspace,
                project_name=opts.comet.project_name,
                **COMET_KWARGS,
            )
        elif exp_type == "resume":
            comet_previous_id = find_existing_comet_id(path)
            exp = ExistingExperiment(
                previous_experiment=comet_previous_id, **COMET_KWARGS
            )
        else:
            exp = None

        trainer = cls(opts, exp)
        trainer.setup(inference_only)

        if ckpt is not None and (path / ckpt).exists():
            print("Loading checkpoint :", str(path / ckpt))
            trainer.load(path / ckpt)

        return trainer

    @classmethod
    def debug_trainer(cls, path="./config/opts.yaml", task="discrete"):
        """
        Utility method to quickly get a trainer and debug stuff

        ```
        from aiphysim.trainer import Trainer

        trainer = Trainer.debug_trainer()
        ```
        """
        opts = load_opts(path, task)
        trainer = cls(opts)
        trainer.setup()
        return trainer

    def setup(self, inference_only=False):
        """
        Set the trainer up. Those functions are basically an initialization but
        having it as a separate function allows for intermediate debugging manipulations
        """
        # Create data loaders
        if not inference_only:
            self.loaders = create_dataloaders(self.opts)

        # set input dim based on data formatting
        if "input_dim" not in self.opts:
            if hasattr(self, "loaders"):
                self.opts.input_dim = self.loaders["train"].dataset.input_dim
            else:
                raise ValueError(
                    "Setup Error: cannot setup a trainer with "
                    + "no loaders and no `input_dim` in its opts"
                )

        # find device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU:", self.device)
        else:
            self.device = torch.device("cpu")
            print("No GPU -> using CPU:", self.device)

        # create NN
        self.model = create_model(self.opts).to(self.device)
        print("\nModel parameters use", mem_size(self.model))
        print(num_params(self.model))

        # create loss util
        self.losses = get_loss(self.opts)

        if not inference_only:
            # create logger to abstract prints away from the main code
            self.logger = Logger(
                self.opts,
                self.exp,
                n_train=len(self.loaders["train"]),
                n_val=len(self.loaders["val"]),
                n_test=len(self.loaders["test"]),
            )
            # create optimizer from opts
            self.optimizer, self.scheduler = get_optimizer(self.opts, self.model)

        if inference_only:
            print(
                "/!\\ Trainer created for inference only:",
                "no logger, no data-loaders, no optimizer",
            )

        # trainer is good to go
        self.is_setup = True

        return self  # enable chain calls

    def update_early_stopping(self, val_loss):
        score = -val_loss

        if self.early_score is None:
            self.early_score = score
            self.save(loss=val_loss)
        elif score < self.early_score + self.opts.early.min_delta:
            self.early_counter += 1
            print(
                "\nEarlyStopping counter: {} out of {}".format(
                    self.early_counter, self.opts.early.patience
                )
            )
            if self.early_counter >= self.opts.early.patience:
                self.early_stop = True
        else:
            self.early_score = score
            self.save(loss=val_loss)
            self.early_counter = 0

    def dev_batch(self, mode="train"):
        """
        Utility class to quickly get a batch
        """
        assert self.is_setup
        batch = next(iter(self.loaders[mode]))
        batch = batch.reshape(-1, batch.shape[0], self.opts.input_dim)
        return batch.to(self.device)

    def run_step(self, batch):
        """
        Execute a training step:
        zero grad -> forward -> loss -> backward -> step -> print

        Args:
            batch (torch.Tensor): Data: batch batch_size x sequence_length x dim
        """
        self.optimizer.zero_grad()

        state = batch.reshape(-1, batch.shape[0], self.opts.input_dim)
        predictions = self.model.forward(state)

        train_losses = self.losses.compute(state, predictions, self.model)
        train_losses["total"].backward()

        self.optimizer.step()

        self.logger.global_step += 1
        self.logger.log_step(train_losses)

    def run_epoch(self):
        """
        Iterate over the entire train data-loader
        """
        self.model.train()
        for self.logger.batch_id, batch in enumerate(self.loaders["train"]):
            batch = batch.to(self.device)
            self.run_step(batch)

    def train(self):
        """
        Execute full training loop based on:
            opts.epochs
            opts.batch_size
        """
        assert self.is_setup
        print("\n>>> Starting training")
        epochs = range(self.logger.epoch_id, self.logger.epoch_id + self.opts.epochs)

        best_val_loss = np.inf
        for self.logger.epoch_id in epochs:

            # train for 1 epoch
            self.run_epoch()

            # evaluate model
            val_loss = self.run_evaluation()

            # update scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(name="best.ckpt")

            # update early-stopping using the val_loss criterion
            self.update_early_stopping(val_loss)  # saves if improvement

            # save latest checkpoint anyway
            self.save()
            print()

            if self.early_stop:
                print("\nEarlyStopping: Patience exceeded, stopping training\n")
                break

        if self.opts.input_dim == 2:
            val_trajs_len = (
                self.opts.val_trajectories.length
                if self.opts.val_trajectories.length > 0
                else self.opts.sequence_length
            )
            batch = next(iter(self.loaders["val"]))
            batch = batch[: self.opts.val_trajectories.n].to(self.device)

            best_state = torch.load(str(self.opts.output_path / "best.ckpt"))
            self.model.load_state_dict(best_state["model_state_dict"])
            plot_2D_comparative_trajectories(
                self.model,
                batch,
                val_trajs_len,
                self.opts.val_trajectories.n_per_plot,
                self.exp,
                self.opts.output_path / "final_traj_plot.png",
            )

    @torch.no_grad()
    def run_evaluation(self):
        """
        TODO: evaluate the model

        Returns:
            float: validation loss
        """
        self.model.eval()
        losses = None
        print()
        for batch in tqdm(self.loaders["val"]):
            batch = batch.to(self.device)

            state = batch.reshape(-1, batch.shape[0], self.opts.input_dim)
            predictions = self.model.forward(state)

            val_losses = self.losses.compute(state, predictions, self.model)
            if losses is None:
                losses = {k: [] for k in val_losses}
            for k, v in val_losses.items():
                losses[k].append(v)
        losses = {k: np.mean([lv.cpu().item() for lv in v]) for k, v in losses.items()}
        self.logger.log_step(losses, mode="val")
        print()
        return losses["total"]

    def save(self, loss=None, name=None):

        # if epoch is not provided just save as "latest.ckpt"
        if loss is None and name is None:
            name = "latest.ckpt"
        elif name is None:
            name = "epoch_{}_loss_{:.4f}.ckpt".format(
                str(self.logger.epoch_id).zfill(3), loss
            )
        else:
            # name is given
            pass

        clean_checkpoints(self.opts.output_path, n_max=5)

        torch.save(
            {
                "early": {
                    "stop": self.early_stop,
                    "score": self.early_score,
                    "counter": self.early_counter,
                },
                "logger": {
                    "global_step": self.logger.global_step,
                    "epoch_id": self.logger.epoch_id,
                    "val_loss": loss,
                },
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler
                else None,
            },
            str(self.opts.output_path / name),
        )

    def load(self, path):
        path = resolve(path)

        state_dict = torch.load(str(path))

        if hasattr(self, "logger"):
            self.logger.global_step = state_dict["logger"]["global_step"]
            self.logger.epoch_id = state_dict["logger"]["epoch_id"]

        self.early_stop = state_dict["early"]["stop"]
        self.early_score = state_dict["early"]["score"]
        self.early_counter = state_dict["early"]["counter"]

        self.model.load_state_dict(state_dict["model_state_dict"])

        if hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        if hasattr(self, "scheduler") and self.scheduler:
            self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
