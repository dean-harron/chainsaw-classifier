# Chainsaw Audio Classifier

A convolutional neural network that detects **chainsaw sounds** in audio recordings. This repo is a production-ready Python package designed to help **detect and curb illegal logging** in remote forests.

> **Mission:** Automated acoustic monitoring for illegal lumbering in protected areas — feed the model an audio clip and decide whether someone is actively cutting trees.

---

## Features

-  **Audio → Mel-spectrogram** feature pipeline powered by `librosa`
-  **3-block CNN** (~110k params) trained end-to-end on log-mel features
-  Clean **Python package** layout (no notebook magic, no Colab-only paths)
-  Configurable via CLI flags **and** a typed config object
-  Checkpointed weights, JSON metrics, and loss-curve plots on every training run
-  `pytest` smoke tests + feature tests
-  Console scripts: `chainsaw-train` and `chainsaw-predict`

---

##  Project Structure

```
chainsaw-classifier/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── data/                          # (you provide) audio samples
│   ├── chainsaw/
│   └── non-chainsaw/
├── models/checkpoints/            # trained .pth weights live here
├── runs/                          # training artifacts (plots, metrics)
├── src/chainsaw_classifier/
│   ├── __init__.py
│   ├── config.py                  # AudioConfig / TrainingConfig / PathConfig
│   ├── features.py                # Mel-spectrogram pipeline
│   ├── dataset.py                 # PyTorch Dataset + padding collate
│   ├── model.py                   # AudioCNN
│   ├── train.py                   # Training loop
│   ├── evaluate.py                # Loss / accuracy / confusion matrix
│   ├── inference.py               # Single-file prediction
│   ├── cli.py                     # `chainsaw-train` entry point
│   └── predict.py                 # `chainsaw-predict` entry point
└── tests/
    ├── conftest.py
    ├── test_smoke.py
    └── test_features.py
```

---

##  Quick Start

### 1. Install

```bash
git clone https://github.com/dean-harron/chainsaw-classifier.git
cd chainsaw-classifier

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

This installs the `chainsaw_classifier` package plus two console scripts: `chainsaw-train` and `chainsaw-predict`.

### 2. Prepare your data

Organize audio files into two folders:

```
data/
└── audio/
    ├── chainsaw/      ← positive class (label = 1)
    │   ├── 001.wav
    │   └── ...
    └── non-chainsaw/  ← negative class (label = 0)
        ├── forest_birds.wav
        └── ...
```

Supported formats: `.wav`, `.mp3`, `.flac`.

### 3. Train

```bash
chainsaw-train --data-dir ./data/audio --epochs 20 --batch-size 16
```

Artifacts written:

| Path | Contents |
|------|----------|
| `models/checkpoints/audio_classifier.pth` | Trained weights |
| `runs/loss_curves.png` | Train/val loss plot |
| `runs/metrics.json` | Final loss, accuracy, confusion matrix |

CLI options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `./data/audio` | Root folder with class subdirs |
| `--epochs` | `10` | Number of epochs |
| `--batch-size` | `16` | Batch size |
| `--learning-rate` | `1e-3` | Adam learning rate |
| `--checkpoint` | `./models/checkpoints/audio_classifier.pth` | Where to save weights |
| `--output-dir` | `./runs` | Where to save plots + metrics |

### 4. Predict

```bash
chainsaw-predict path/to/clip.wav
#  No chainsaw detected  (Non-Chainsaw) in clip.wav

chainsaw-predict path/to/clip.wav --quiet
# Non-Chainsaw
```

Exit codes: `0` always; use `--quiet` in scripts and parse the printed label.

### 5. Programmatic use

```python
from chainsaw_classifier.config import CONFIG
from chainsaw_classifier.inference import load_model, predict_audio

model = load_model(CONFIG.paths.default_checkpoint, CONFIG)
label = predict_audio(model, "data/chainsaw/example.wav", CONFIG)
print(label)  # "Chainsaw" or "Non-Chainsaw"
```

---

##  How it Works

1. **Load** audio at 22,050 Hz, pad/truncate to 4 seconds.
2. Compute a **64-band log-Mel spectrogram** with `n_fft=1024`, `hop_length=512`.
3. Standardize (zero mean, unit variance).
4. Feed the `(1, 64, T)` tensor to the **AudioCNN**:

   ```
   Conv(1→16) → BN → ReLU → MaxPool(2)
   Conv(16→32) → BN → ReLU → MaxPool(2)
   Conv(32→64) → BN → ReLU → MaxPool(2)
   Dropout(0.3)
   Linear(flatten → 128) → ReLU → Linear(128 → 2)
   ```

5. The model outputs class logits; argmax → `Chainsaw` / `Non-Chainsaw`.

### Why Mel-spectrograms?

Chainsaws produce distinctive harmonic and broadband noise patterns that map cleanly onto the time-frequency plane. CNNs trained on Mel features generalize well across recording equipment and environments.

---

##  Use Cases

-  **Forest ranger alerts** — passive acoustic monitoring at the edge
-  **Acoustic sensor networks** — flag chainsaw events in real time
-  **Supply-chain auditing** — screen audio evidence of illegal harvesting
-  **Research baseline** — reproducible starting point for environmental audio ML

---

##  Testing

```bash
pip install -e .[dev]
pytest
```

Tests run without GPU, without audio files, and without any model weights.

---

##  Roadmap

- [ ] Replace fixed FC flatten size with adaptive pooling
- [ ] Add data augmentation (time-stretch, pitch-shift, mixup)
- [ ] Real-time streaming inference (chunked audio)
- [ ] ONNX export for edge deployment
- [ ] Pre-trained checkpoints for common forest datasets

---

##  License

Released under the [MIT License](LICENSE). © 2026 Dean Harron.

---

##  Acknowledgements

- [librosa](https://librosa.org/) — audio feature extraction
- [PyTorch](https://pytorch.org/) — deep learning framework
- [scikit-learn](https://scikit-learn.org/) — evaluation metrics

> **Conservation impact:** Every illegal logging event caught early is a habitat preserved. If this tool helps your project, please star the repo and share your results.
