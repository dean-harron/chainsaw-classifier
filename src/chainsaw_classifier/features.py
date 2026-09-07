"""Audio feature extraction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import librosa
import numpy as np
import torch

from .config import AudioConfig

PathLike = Union[str, Path]


def load_audio(
    filepath: PathLike,
    sample_rate: int,
    duration_seconds: float,
) -> np.ndarray:
    signal, _ = librosa.load(str(filepath), sr=sample_rate)
    max_length = int(sample_rate * duration_seconds)
    if len(signal) > max_length:
        return signal[:max_length]
    pad_width = max_length - len(signal)
    return np.pad(signal, (0, pad_width), mode="constant")


def compute_mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mean = mel_spec_db.mean()
    std = mel_spec_db.std() + 1e-6
    return (mel_spec_db - mean) / std


def signal_to_tensor(
    signal: np.ndarray,
    audio_cfg: AudioConfig,
) -> torch.Tensor:
    mel_spec = compute_mel_spectrogram(
        signal=signal,
        sample_rate=audio_cfg.sample_rate,
        n_mels=audio_cfg.n_mels,
        n_fft=audio_cfg.n_fft,
        hop_length=audio_cfg.hop_length,
    )
    return torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
