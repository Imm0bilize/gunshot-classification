import hydra
import pytorch_lightning as pl

import torchmetrics


class ModelWrapper(pl.LightningModule):
    def __init__(self, config, is_need2rgb: bool):
        super().__init__()
        self.config = config
        self._model = hydra.utils.instantiate(self.config.models)
        self.print(f"Instantiated model <{self._model.__class__.__name__}>")
        self._loss_fn = hydra.utils.instantiate(self.config.losses)
        self.print(f"Instantiated loss function <{self._loss_fn.__class__.__name__}>")

        self.train_accuracy = torchmetrics.Accuracy()
        self.loss_accurancy = torchmetrics.Accuracy()

    def forward(self, x):
        if self.is_need2rgb:
            pass
        return self._model(x)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optims,
            params=self._model.parameters(),
            lr=self.config.hyperparams.lr,
        )
        self.print(f"Instantiated optimizers <{optimizer.__class__.__name__}>")
        # print(f"Instantiated optimizers <{optimizer.__class__.__name__}>")
        return optimizer

    # ----------------- TRAIN -----------------
    def training_step(self, batch, batch_idx):
        x, y = batch

        prediction = self.forward(x)
        loss = self._loss_fn(prediction, y)
        self.train_accuracy(prediction, y)
        self.log(f"train_loss", loss, on_step=True)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc',  self.train_accuracy.compute())
        self.train_accuracy.reset()

    # ----------------- VAL -----------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.forward(x)
        loss = self._loss_fn(prediction, y)
        self.val_accuracy.update(prediction, y)
        self.log(f"val_loss", loss, on_step=True)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_accuracy.compute())
        self.val_accuracy.reset()
