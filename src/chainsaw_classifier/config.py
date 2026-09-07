"""Project-wide configuration constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    duration_seconds: float = 4.0
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-3
    num_epochs: int = 10
    train_split: float = 0.9
    seed: int = 42
    num_workers: int = 0
    dropout: float = 0.3
    num_classes: int = 2
    label_names: Tuple[str, ...] = ("non-chainsaw", "chainsaw")


@dataclass(frozen=True)
class PathConfig:
    data_dir: Path = PROJECT_ROOT / "data" / "audio"
    model_dir: Path = PROJECT_ROOT / "models" / "checkpoints"
    artifacts_dir: Path = PROJECT_ROOT / "runs"
    default_checkpoint: Path = PROJECT_ROOT / "models" / "checkpoints" / "audio_classifier.pth"
    log_dir: Path = PROJECT_ROOT / "runs" / "logs"


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)


CONFIG = AppConfig()


CHAINSAW_LABEL = 1
NON_CHAINSAW_LABEL = 0
CLASS_LABELS: Tuple[str, str] = ("Non-Chainsaw", "Chainsaw")
