"""Command-line entry point for training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt

from chainsaw_classifier.config import CONFIG, AppConfig, AudioConfig, TrainingConfig  # noqa: E402
from chainsaw_classifier.dataset import ChainSawAudioDataset  # noqa: E402
from chainsaw_classifier.evaluate import evaluate_model  # noqa: E402
from chainsaw_classifier.model import AudioCNN  # noqa: E402
from chainsaw_classifier.train import build_dataloaders, save_checkpoint, set_seed, train_model  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chainsaw-train",
        description="Train the chainsaw audio classifier.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=CONFIG.paths.data_dir,
        help="Root directory containing 'chainsaw/' and 'non-chainsaw/' subfolders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG.training.num_epochs,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CONFIG.training.batch_size,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=CONFIG.training.learning_rate,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CONFIG.paths.default_checkpoint,
        help="Where to save the trained model weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONFIG.paths.artifacts_dir,
        help="Directory to write loss curves and metrics.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = AppConfig(
        audio=AudioConfig(),
        training=TrainingConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        ),
        paths=CONFIG.paths,
    )

    set_seed(cfg.training.seed)

    print(f"Loading dataset from {args.data_dir} ...")
    dataset = ChainSawAudioDataset(args.data_dir, audio_cfg=cfg.audio)
    print(f"Total samples: {len(dataset)}")

    train_loader, test_loader = build_dataloaders(dataset, cfg)
    print(f"Train: {len(train_loader.dataset)}  Test: {len(test_loader.dataset)}")

    model = AudioCNN(audio_cfg=cfg.audio, training_cfg=cfg.training)
    print(model)

    result = train_model(model, train_loader, test_loader, cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_plot_path = args.output_dir / "loss_curves.png"
    _plot_losses(result.train_losses, result.val_losses, loss_plot_path)
    print(f"Loss curves saved to {loss_plot_path}")

    eval_result = evaluate_model(model, test_loader, cfg)
    metrics = {
        "validation_loss": eval_result.loss,
        "validation_accuracy": eval_result.accuracy,
        "confusion_matrix": eval_result.confusion_matrix,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(
        f"Validation Loss: {eval_result.loss:.4f}  "
        f"Accuracy: {eval_result.accuracy:.4f}"
    )
    print(f"Metrics saved to {metrics_path}")

    save_checkpoint(model, args.checkpoint)
    print(f"Model checkpoint saved to {args.checkpoint}")

    return 0


def _plot_losses(train_losses: list[float], val_losses: list[float], out_path: Path) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    sys.exit(main())
