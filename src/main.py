import hydra
import wandb
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig

from src.model_wrapper import ModelWrapper
from src.utils import get_wandb_token
from src.dataset import GunShotDataset


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    token = get_wandb_token()
    torch.manual_seed(cfg.hyperparams.seed)
    seed_everything(cfg.hyperparams.seed)

    wandb.login(key=token)
    wandb_logger = WandbLogger(project='GunShotClassificationV2',)

    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')
    wrapped_model = ModelWrapper(cfg, False)

    train_dataloader = DataLoader(
        dataset=GunShotDataset(),
        batch_size=10,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=GunShotDataset(),
        batch_size=10,
        shuffle=False,
    )

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=5
    )

    trainer.fit(wrapped_model, train_dataloader, val_dataloader)
    wandb.finish()


if __name__ == '__main__':
    main()
