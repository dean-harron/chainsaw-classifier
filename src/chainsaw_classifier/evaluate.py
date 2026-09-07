"""Model evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from .config import CONFIG, TrainingConfig
from .model import AudioCNN


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    confusion_matrix: List[List[int]]


def evaluate_model(
    model: AudioCNN,
    test_loader: DataLoader,
    cfg=CONFIG,
) -> EvaluationResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(test_loader), 1)
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(cfg.training.num_classes)))

    return EvaluationResult(loss=avg_loss, accuracy=accuracy, confusion_matrix=cm.tolist())
