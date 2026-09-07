"""Inference helpers for single-file prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from .config import CHAINSAW_LABEL, CLASS_LABELS, AudioConfig, CONFIG
from .features import compute_mel_spectrogram, load_audio
from .model import AudioCNN

PathLike = Union[str, Path]


def load_model(checkpoint_path: PathLike, cfg=CONFIG) -> AudioCNN:
    model = AudioCNN(audio_cfg=cfg.audio, training_cfg=cfg.training)
    state_dict = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_audio(
    model: AudioCNN,
    filepath: PathLike,
    cfg=CONFIG,
) -> str:
    audio_cfg: AudioConfig = cfg.audio
    signal = load_audio(
        filepath,
        sample_rate=audio_cfg.sample_rate,
        duration_seconds=audio_cfg.duration_seconds,
    )
    mel_spec = compute_mel_spectrogram(
        signal=signal,
        sample_rate=audio_cfg.sample_rate,
        n_mels=audio_cfg.n_mels,
        n_fft=audio_cfg.n_fft,
        hop_length=audio_cfg.hop_length,
    )
    tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        predicted_idx = int(torch.argmax(outputs, dim=1).item())

    return CLASS_LABELS[predicted_idx]


def is_chainsaw_prediction(predicted_label: str) -> bool:
    return predicted_label == CLASS_LABELS[CHAINSAW_LABEL]
