import torch

from koop.dataloading import create_dataloaders
from koop.logger import Logger
from koop.losses import Loss
from koop.model import DeepKoopman
from koop.utils import make_output_dir, get_optimizer, load_opts


class Trainer:
    def __init__(self, opts) -> None:
        self.opts = opts
        self.is_setup = False

    @classmethod
    def debug_trainer(cls, path="./config/opts.yaml", task="discrete"):
        opts = load_opts(path, task)
        trainer = cls(opts)
        trainer.setup()
        return trainer

    def setup(self):
        self.loaders = create_dataloaders(
            self.opts["data"], self.opts["sequence_length"], self.opts["batch_size"]
        )
        self.opts.input_dim = self.loaders["train"].dataset.input_dim

        self.opts.output_path = make_output_dir(
            self.opts.output_path, dev=self.opts.get("dev")
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU:", self.device)
        else:
            self.device = torch.device("cpu")
            print("No GPU -> using CPU:", self.device)

        self.model = DeepKoopman(self.opts).to(self.device)
        self.losses = Loss(self.opts)
        self.logger = Logger(
            self.opts,
            n_train=len(self.loaders["train"]),
            n_val=len(self.loaders["val"]),
            n_test=len(self.loaders["test"]),
        )
        self.optimizer, self.scheduler = get_optimizer(self.opts, self.model)

        self.is_setup = True

    def dev_batch(self, mode="train"):
        assert self.is_setup
        return next(iter(self.loaders[mode])).to(self.device)

    def run_step(self, batch):
        self.optimizer.zero_grad()

        state = batch[:, 0, :]
        predictions = self.model.forward(state)

        train_losses = self.losses.compute(batch, predictions, self.model)
        train_losses["total"].backward()

        self.optimizer.step()

        self.logger.global_step += 1
        self.logger.print_step(train_losses)

    def run_epoch(self):
        self.model.train()
        for self.logger.batch_id, batch in enumerate(self.loaders["train"]):
            batch = batch.to(self.device)
            self.run_step(batch)

    def train(self):
        assert self.is_setup
        for self.logger.epoch_id in range(self.opts.epochs):
            self.run_epoch()
            val_loss = self.run_evaluation()
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            self.save()

    def run_evaluation(self):
        self.model.eval()
        val_loss = torch.tensor(0.0)
        return val_loss

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
