import torch

from koop.dataloading import create_dataloaders
from koop.model import DeepKoopman
from koop.losses import Loss
from koop.logger import Logger


class Trainer:
    def __init__(self, opts) -> None:
        self.opts = opts

    def setup(self):
        self.dataloaders = create_dataloaders(
            self.opts["data"], self.opts["sequence_length"], self.opts["batch_size"]
        )
        self.opts.input_dim = self.dataloaders["train"].dataset.input_dim

        self.model = DeepKoopman(self.opts)
        self.losses = Loss(self.opts)
        self.logger = Logger(self.opts)

    def run_step(self, batch):
        self.logger.global_step += 1
        self.logger.print_step()

    def run_epoch(self):
        for self.logger.batch_id, batch in enumerate(self.dataloaders["train"]):
            self.run_step(batch)

    def train(self):
        for self.logger.epoch_id in range(self.opts.epochs):
            self.run_epoch()
            self.run_evaluation()
            self.save()

    def run_evaluation(self):
        # todo
        pass

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
