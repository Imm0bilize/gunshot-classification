import sys

import hydra
import pytorch_lightning as pl
import torch

import torchmetrics


class ModelWrapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self._model = hydra.utils.instantiate(self.config.models).to(self.dev)

        print(f"Instantiated model <{self._model.__class__.__name__}>", file=sys.stderr)
        self._loss_fn = hydra.utils.instantiate(self.config.losses)
        print(f"Instantiated loss function <{self._loss_fn.__class__.__name__}>", file=sys.stderr)
        self._activation_fn = hydra.utils.instantiate(self.config.activation_fn)
        print(f"Instantiated activation function <{self._activation_fn.__class__.__name__}>", file=sys.stderr)

        self.train_accuracy = torchmetrics.classification.BinaryAccuracy().to(self.dev)
        self.val_accuracy = torchmetrics.classification.BinaryAccuracy().to(self.dev)

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optims,
            params=self._model.parameters(),
            lr=self.config.hyperparams.lr,
        )
        print(f"Instantiated optimizers <{optimizer.__class__.__name__}>", file=sys.stderr)
        return optimizer

    # ----------------- TRAIN -----------------
    def training_step(self, batch, batch_idx):
        x, y = batch

        x, y = x.to(self.dev), y.to(self.dev)

        prediction = self._activation_fn(self.forward(x))

        loss = self._loss_fn(prediction, y)

        self.log("train_loss", loss.item(), on_epoch=True)

        self.train_accuracy(prediction, y)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_accuracy.compute())
        self.train_accuracy.reset()

    # ----------------- VAL -----------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.dev), y.to(self.dev)

        prediction = self._activation_fn(self.forward(x))
        loss = self._loss_fn(prediction, y)
        self.log("val_loss", loss.item(), on_epoch=True)

        self.val_accuracy(prediction, y)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_accuracy.compute())
        self.val_accuracy.reset()
