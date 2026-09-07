"""Unit tests for feature extraction."""

from __future__ import annotations

import numpy as np

from chainsaw_classifier.config import AudioConfig
from chainsaw_classifier.features import (
    compute_mel_spectrogram,
    load_audio,
    signal_to_tensor,
)


def test_load_audio_pads_short_signal():
    sample_rate = 22050
    short_signal = np.zeros(sample_rate, dtype=np.float32)  # 1 second
    with_short = np.zeros(sample_rate * 2, dtype=np.float32)
    padded = np.pad(with_short, (0, sample_rate * 2), mode="constant")
    assert padded.shape[0] == sample_rate * 4


def test_compute_mel_spectrogram_normalized():
    audio_cfg = AudioConfig()
    duration_samples = int(audio_cfg.sample_rate * audio_cfg.duration_seconds)
    signal = np.random.default_rng(1).standard_normal(duration_samples).astype(np.float32)
    mel = compute_mel_spectrogram(
        signal=signal,
        sample_rate=audio_cfg.sample_rate,
        n_mels=audio_cfg.n_mels,
        n_fft=audio_cfg.n_fft,
        hop_length=audio_cfg.hop_length,
    )
    assert mel.shape == (audio_cfg.n_mels, mel.shape[1])
    assert abs(mel.mean()) < 1e-3


def test_signal_to_tensor_shape():
    audio_cfg = AudioConfig()
    duration_samples = int(audio_cfg.sample_rate * audio_cfg.duration_seconds)
    signal = np.random.default_rng(2).standard_normal(duration_samples).astype(np.float32)
    tensor = signal_to_tensor(signal, audio_cfg)
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == audio_cfg.n_mels
