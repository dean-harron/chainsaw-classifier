"""Convolutional neural network for mel-spectrogram classification."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import AudioConfig, TrainingConfig
from .dataset import compute_flatten_size


class AudioCNN(nn.Module):
    """3-block CNN operating on log-mel spectrograms."""

    def __init__(
        self,
        audio_cfg: AudioConfig | None = None,
        training_cfg: TrainingConfig | None = None,
    ) -> None:
        super().__init__()
        audio_cfg = audio_cfg or AudioConfig()
        training_cfg = training_cfg or TrainingConfig()

        self.audio_cfg = audio_cfg
        self.training_cfg = training_cfg

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(training_cfg.dropout)

        flatten_size = compute_flatten_size(audio_cfg)
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, training_cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)
