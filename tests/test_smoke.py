"""Smoke tests for the chainsaw classifier package."""

from __future__ import annotations

import numpy as np
import torch

from chainsaw_classifier.config import AudioConfig, CONFIG
from chainsaw_classifier.dataset import compute_flatten_size
from chainsaw_classifier.features import (
    compute_mel_spectrogram,
    load_audio,
    signal_to_tensor,
)
from chainsaw_classifier.inference import load_model
from chainsaw_classifier.model import AudioCNN


def test_audio_config_defaults():
    assert CONFIG.audio.sample_rate == 22050
    assert CONFIG.audio.n_mels == 64
    assert CONFIG.audio.duration_seconds == 4.0


def test_signal_to_tensor_shape():
    audio_cfg = AudioConfig()
    duration_samples = int(audio_cfg.sample_rate * audio_cfg.duration_seconds)
    signal = np.random.default_rng(0).standard_normal(duration_samples).astype(np.float32)
    tensor = signal_to_tensor(signal, audio_cfg)
    assert tensor.ndim == 3
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == audio_cfg.n_mels


def test_compute_flatten_size_matches_model():
    audio_cfg = AudioConfig()
    flatten = compute_flatten_size(audio_cfg)
    model = AudioCNN(audio_cfg=audio_cfg)
    first_linear = next(m for m in model.classifier if isinstance(m, torch.nn.Linear))
    assert first_linear.in_features == flatten


def test_model_forward_shape():
    audio_cfg = AudioConfig()
    model = AudioCNN(audio_cfg=audio_cfg)
    batch = torch.randn(2, 1, audio_cfg.n_mels, 173)
    logits = model(batch)
    assert logits.shape == (2, CONFIG.training.num_classes)


def test_load_model_constructs_with_random_weights():
    model = load_model_checkpoint_random()
    assert isinstance(model, AudioCNN)


def model_with_random_weights() -> AudioCNN:
    audio_cfg = AudioConfig()
    model = AudioCNN(audio_cfg=audio_cfg)
    return model


def load_model_checkpoint_random() -> AudioCNN:
    import tempfile

    audio_cfg = AudioConfig()
    model = model_with_random_weights()
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(model.state_dict(), tmp.name)
        loaded = load_model(tmp.name, CONFIG)
    assert isinstance(loaded, AudioCNN)
    return loaded
