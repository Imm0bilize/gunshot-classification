import hydra
import torch
import torchaudio
import numpy as np
from glob import glob
from torch.utils.data import Dataset

from src.utils import AugmentationNoise


class GunShotDataset(Dataset):
    def __init__(self, paths_to_audio, augmentation_cfg, preproc_cfg, is_train, augmentation_noize=None):
        assert len(paths_to_audio), "The list of audio paths is empty"
        self.paths_to_audio = paths_to_audio

        self.augmentation_cfg = augmentation_cfg
        self.preproc_cfg = preproc_cfg
        self.is_train = is_train
        self.eps = 1e-9

        if is_train:
            # assert augmentation_noize, "Train mode, but augmentation noize is None"
            self.aug_noize = None
            augmentation = [
                torchaudio.transforms.FrequencyMasking(
                    freq_mask_param=self.augmentation_cfg.freq_mask_param
                ),
                torchaudio.transforms.TimeMasking(
                    time_mask_param=self.augmentation_cfg.time_mask_param
                )
            ]
        else:
            augmentation = []

        self._preprocessing = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.preproc_cfg.mel_params.sample_rate,
                n_mels=self.preproc_cfg.mel_params.n_mels,
                n_fft=self.preproc_cfg.mel_params.n_fft,
                hop_length=self.preproc_cfg.mel_params.hop_length,
                f_max=self.preproc_cfg.mel_params.f_max,
                f_min=self.preproc_cfg.mel_params.f_min,
            ),

            *augmentation,
        )

    def __len__(self):
        return len(self.paths_to_audio)

    def _apply_augmentation(self, wave, sr):
        # TODO: fix augmentation noize error
        # if np.random.rand() > 1.0 - self.augmentation_cfg.noize.probability:
        #     alpha = np.random.randint(*self.augmentation_cfg.noize.volume_range) / 100
        #     wave = self.aug_noize(wave, alpha)
        if np.random.rand() > 1.0 - self.augmentation_cfg.gain.probability:
            power = np.random.randint(*self.augmentation_cfg.gain.distortion_range) / 10
            wave = torchaudio.functional.gain(wave, power)
        if np.random.rand() > 1.0 - self.augmentation_cfg.highpass.probability:
            freq = np.random.randint(*self.augmentation_cfg.highpass.cutoff_range)
            wave = torchaudio.functional.highpass_biquad(wave, sr, cutoff_freq=freq)
        if np.random.rand() > 1.0 - self.augmentation_cfg.lowpass.probability:
            freq = np.random.randint(*self.augmentation_cfg.lowpass.cutoff_range)
            wave = torchaudio.functional.lowpass_biquad(wave, sr, cutoff_freq=freq)
        return wave

    def _normalize(self, spec: torch.Tensor) -> torch.Tensor:
        mean = spec.mean()
        std = spec.std()
        return (spec - mean) / (std + self.eps)

    @staticmethod
    def _stereo2mono(x):
        if x.shape[0] == 1:
            return x
        x = torch.mean(x, dim=0)
        x = torch.unsqueeze(x, dim=0)
        return x

    def __getitem__(self, idx):
        wave, sr = torchaudio.load(filepath=self.paths_to_audio[idx], normalize=True)
        wave = self._stereo2mono(wave)
        if self.is_train:
            wave = self._apply_augmentation(wave, sr)

        image = torch.zeros(1, self.preproc_cfg.mel_params.n_mels, self.preproc_cfg.img_padding_length)

        mel_spectrogram = self._normalize(torch.log(self._preprocessing(wave) + self.eps))
        image[0, :, :mel_spectrogram.size(2)] = mel_spectrogram[:, :, :self.preproc_cfg.img_padding_length]

        y = torch.tensor(1, dtype=torch.float32) if self.paths_to_audio[idx].find('not') == -1 \
            else torch.tensor(0, dtype=torch.float32)
        return image, torch.unsqueeze(y, -1)


@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    files = sorted(glob(""))

    ds = GunShotDataset(files, cfg.augmentations, cfg.preprocessing, True, AugmentationNoise(""))

    for i in range(len(ds)):
        x, y = ds[i]
        print(x.shape, y)


if __name__ == '__main__':
    main()
