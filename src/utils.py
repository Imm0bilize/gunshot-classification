import os
import random
from glob import glob
from contextlib import contextmanager

import wandb
import torch
import torchaudio

from dotenv import load_dotenv


class AugmentationNoise(torch.nn.Module):
    def __init__(self, path_to_aug_sound, alpha=0.1):
        super(AugmentationNoise, self).__init__()

        audio, sr = torchaudio.load(path_to_aug_sound)
        self.audio = torch.unsqueeze(torch.mean(torchaudio.functional.resample(audio, sr, 16000), dim=0), dim=0)
        self._noise_audio = audio.sum(dim=0)

    def forward(self, wav, alpha):
        # return wav + alpha * self._noise_audio[:wav.shape[-1]]
        return wav + alpha * self.audio[:wav.shape[-1]]


def get_wandb_token():
    load_dotenv()
    try:
        return os.environ["WANDB_TOKEN"]
    except KeyError:
        raise RuntimeError("Environ does not have wandb token!")


def validation_split(*args, percent=0.3):
    train_set = []
    val_set = []
    [random.shuffle(files) for files in args]
    for class_labels in args:
        label_length = len(class_labels)
        val_set.extend(class_labels[:int(label_length * percent)])
        train_set.extend(class_labels[int(label_length * percent):])
    return train_set, val_set


def save_artifact(path_to_weights):
    artifact = wandb.Artifact(name="model weights", type="model weights")
    artifact.add_file(path_to_weights)
    wandb.run.log_artifact(artifact)
    wandb.run.join()
