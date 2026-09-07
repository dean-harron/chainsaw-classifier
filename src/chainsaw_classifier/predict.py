"""Command-line entry point for inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chainsaw_classifier.config import CONFIG  # noqa: E402
from chainsaw_classifier.inference import is_chainsaw_prediction, load_model, predict_audio  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chainsaw-predict",
        description="Run chainsaw audio classification on an audio file.",
    )
    parser.add_argument("audio_path", type=Path, help="Path to a .wav/.mp3/.flac file.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CONFIG.paths.default_checkpoint,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the prediction label (no extra info).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.audio_path.is_file():
        print(f"Error: audio file not found: {args.audio_path}", file=sys.stderr)
        return 1
    if not args.checkpoint.is_file():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    model = load_model(args.checkpoint, CONFIG)
    label = predict_audio(model, args.audio_path, CONFIG)

    if args.quiet:
        print(label)
    else:
        chainsaw = is_chainsaw_prediction(label)
        verdict = "⚠️  Chainsaw detected" if chainsaw else "✅ No chainsaw detected"
        print(f"{verdict}  ({label}) in {args.audio_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
