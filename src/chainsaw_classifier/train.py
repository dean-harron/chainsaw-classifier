"""Training loop for the chainsaw audio classifier."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chainsaw_classifier.config import AppConfig, CONFIG  # noqa: E402
from chainsaw_classifier.dataset import ChainSawAudioDataset, collate_padded  # noqa: E402
from chainsaw_classifier.model import AudioCNN  # noqa: E402


@dataclass
class TrainResult:
    train_losses: list[float]
    val_losses: list[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    dataset: ChainSawAudioDataset,
    cfg: AppConfig = CONFIG,
) -> Tuple[DataLoader, DataLoader]:
    train_size = int(cfg.training.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_subset, test_subset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_padded,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_padded,
    )
    return train_loader, test_loader


def train_model(
    model: AudioCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: AppConfig = CONFIG,
) -> TrainResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(cfg.training.num_epochs):
        model.train()
        running_train = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train += loss.item()
        epoch_train_loss = running_train / max(len(train_loader), 1)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val += loss.item()
        epoch_val_loss = running_val / max(len(val_loader), 1)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch [{epoch + 1}/{cfg.training.num_epochs}] "
            f"Train Loss: {epoch_train_loss:.4f} "
            f"Val Loss: {epoch_val_loss:.4f}"
        )

    return TrainResult(train_losses=train_losses, val_losses=val_losses)


def save_checkpoint(model: AudioCNN, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    from chainsaw_classifier.cli import main

    sys.exit(main())
