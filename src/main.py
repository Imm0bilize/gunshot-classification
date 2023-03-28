import os
from glob import glob
from datetime import datetime

import hydra
import wandb
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.model_wrapper import ModelWrapper
from src.utils import get_wandb_token, save_artifact, validation_split
from src.dataset import GunShotDataset


def create_callbacks(weights_name):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename=weights_name,
        dirpath=hydra.utils.to_absolute_path("checkpoints")
    )

    stopping_callback = EarlyStopping(
        monitor="val_acc",
        patience=20,
        verbose=True,
        mode='max'
    )

    return [checkpoint_callback, stopping_callback]


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # token = get_wandb_token()
    # wandb.login(key=token)
    # wandb_logger = WandbLogger(project='GunShotClassificationV2', log_model=True)

    print(OmegaConf.to_yaml(cfg))

    print(f"Set seed: {cfg.hyperparams.seed}")
    torch.manual_seed(cfg.hyperparams.seed)
    seed_everything(cfg.hyperparams.seed)

    train_paths, val_paths = validation_split(
        glob(os.path.join(hydra.utils.to_absolute_path("datasets/weapon"), "*.wav")),
        glob(os.path.join(hydra.utils.to_absolute_path("datasets/not_weapon"), "*.wav")),
        percent=cfg.hyperparams.val_split
    )

    train_dataloader = DataLoader(
        dataset=GunShotDataset(
            train_paths,
            cfg.augmentations,
            cfg.preprocessing,
            is_train=True,
            is_need2rgb="torchvision" in cfg.models._target_
        ),
        batch_size=cfg.hyperparams.batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_dataloader = DataLoader(
        dataset=GunShotDataset(
            val_paths,
            cfg.augmentations,
            cfg.preprocessing,
            is_train=False,
            is_need2rgb="torchvision" in cfg.models._target_
        ),
        batch_size=cfg.hyperparams.batch_size,
        shuffle=False,
        num_workers=2,
    )
    print("Make model wrapper...")

    wrapped_model = ModelWrapper(cfg)

    weights_name = f"{cfg.models._target_.split('.')[-1]}-{datetime.now().strftime('%d-%m_%H:%M')}"
    trainer = Trainer(
        # gpus=1,
        # logger=wandb_logger,
        callbacks=create_callbacks(weights_name),
        max_epochs=cfg.hyperparams.n_epochs,
        deterministic=True
    )

    trainer.fit(wrapped_model, train_dataloader, val_dataloader)
    # save_artifact(weights_name)
    # wandb.finish()

if __name__ == '__main__':
    main()
