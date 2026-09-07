"""PyTorch dataset for chainsaw audio classification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import AudioConfig
from .features import load_audio, signal_to_tensor

PathLike = Union[str, Path]
Sample = Tuple[torch.Tensor, torch.Tensor]


SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac")


class ChainSawAudioDataset(Dataset):
    """Loads audio files from a folder structure:

    root_dir/
        non-chainsaw/
            *.wav
        chainsaw/
            *.wav
    """

    def __init__(
        self,
        root_dir: PathLike,
        audio_cfg: AudioConfig | None = None,
    ) -> None:
        self.audio_cfg = audio_cfg or AudioConfig()
        self.samples: List[Tuple[str, int]] = self._index_samples(Path(root_dir))
        if not self.samples:
            raise FileNotFoundError(
                f"No audio files found under '{root_dir}'. "
                "Expected subfolders 'chainsaw' and 'non-chainsaw'."
            )

    @staticmethod
    def _index_samples(root_dir: Path) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for label, folder in enumerate(["non-chainsaw", "chainsaw"]):
            folder_path = root_dir / folder
            if not folder_path.is_dir():
                continue
            for entry in sorted(os.listdir(folder_path)):
                if entry.lower().endswith(SUPPORTED_EXTENSIONS):
                    samples.append((str(folder_path / entry), label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        filepath, label = self.samples[idx]
        signal = load_audio(
            filepath,
            sample_rate=self.audio_cfg.sample_rate,
            duration_seconds=self.audio_cfg.duration_seconds,
        )
        mel_tensor = signal_to_tensor(signal, self.audio_cfg)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return mel_tensor, label_tensor


def collate_padded(batch: List[Sample]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad mel-spectrograms along the time axis so a batch has uniform shape."""
    spectrograms, labels = zip(*batch, strict=True)
    max_time = max(spec.shape[-1] for spec in spectrograms)
    padded = torch.zeros(len(spectrograms), spectrograms[0].shape[0], spectrograms[0].shape[1], max_time)
    for i, spec in enumerate(spectrograms):
        padded[i, :, :, : spec.shape[-1]] = spec
    labels_tensor = torch.stack(labels)
    return padded, labels_tensor


def compute_flatten_size(
    audio_cfg: AudioConfig,
    time_frames: int | None = None,
) -> int:
    """Compute the flattened feature size after three 2x max-pools."""
    if time_frames is None:
        time_frames = int(
            audio_cfg.sample_rate * audio_cfg.duration_seconds / audio_cfg.hop_length
        ) + 1
    mel_height = audio_cfg.n_mels
    for _ in range(3):
        mel_height //= 2
        time_frames //= 2
    return 64 * mel_height * time_frames  # 64 channels after conv3
