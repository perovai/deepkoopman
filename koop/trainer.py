import torch
from tqdm import tqdm
import numpy as np

from koop.dataloading import create_dataloaders
from koop.logger import Logger
from koop.losses import Loss
from koop.model import DeepKoopman
from koop.utils import get_optimizer, load_opts, make_output_dir, mem_size, num_params


class Trainer:
    def __init__(self, opts, exp=None) -> None:
        self.opts = opts
        self.is_setup = False
        self.exp = exp

    @classmethod
    def debug_trainer(cls, path="./config/opts.yaml", task="discrete"):
        """
        Utility method to quickly get a trainer and debug stuff

        ```
        from koop.trainer import Trainer

        trainer = Trainer.debug_trainer()
        ```
        """
        opts = load_opts(path, task)
        trainer = cls(opts)
        trainer.setup()
        return trainer

    def setup(self):
        """
        Set the trainer up. Those functions are basically an initialization but
        having it as a separate function allows for intermediate debugging manipulations
        """
        # Create data loaders
        lims = {
            f"{mode}_lim": self.opts.get("limit", {}).get(mode, -1)
            for mode in ["train", "val", "test"]
        }
        self.loaders = create_dataloaders(
            self.opts.data_folder,
            self.opts.sequence_length,
            self.opts.batch_size,
            self.opts.workers,
            **lims,
        )

        # set input dim based on data formatting
        self.opts.input_dim = self.loaders["train"].dataset.input_dim

        # find device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU:", self.device)
        else:
            self.device = torch.device("cpu")
            print("No GPU -> using CPU:", self.device)

        # create NN
        self.model = DeepKoopman(self.opts).to(self.device)
        print("\nModel parameters use", mem_size(self.model))
        print(num_params(self.model))

        # create loss util
        self.losses = Loss(self.opts)

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

        # trainer is good to go
        self.is_setup = True

    def dev_batch(self, mode="train"):
        """
        Utility class to quickly get a batch
        """
        assert self.is_setup
        return next(iter(self.loaders[mode])).to(self.device)

    def run_step(self, batch):
        """
        Execute a training step:
        zero grad -> forward -> loss -> backward -> step -> print

        Args:
            batch (torch.Tensor): Data: batch batch_size x sequence_length x dim
        """
        self.optimizer.zero_grad()

        state = batch[:, 0, :]
        predictions = self.model.forward(state)

        train_losses = self.losses.compute(batch, predictions, self.model)
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
        for self.logger.epoch_id in range(self.opts.epochs):
            self.run_epoch()
            val_loss = self.run_evaluation()
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            self.save()
            print()

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
            state = batch[:, 0, :]
            predictions = self.model.forward(state)
            val_losses = self.losses.compute(batch, predictions, self.model)
            if losses is None:
                losses = {k: [] for k in val_losses}
            for k, v in val_losses.items():
                losses[k].append(v)
        losses = {k: np.mean(v) for k, v in losses.items()}
        self.logger.log_step(losses, mode="val")
        print()
        return losses["total"]

    def save(self):
        return  # todo: add output_path, save_every
        if self.logger.epoch_id % self.opts.save_every:
            torch.save(
                {
                    "model": self.model,
                    "optimizer": self.optimizer,
                },
                self.opts["output_path"],
            )
