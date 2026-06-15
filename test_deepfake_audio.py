"""
Deepfake Audio Detection — Inference Script
============================================
Tests the trained BetterClassifier model on new .wav audio samples.

Usage:
    # Single file
    python test_deepfake_audio.py --audio path/to/audio.wav

    # Folder of files
    python test_deepfake_audio.py --audio path/to/folder/

    # Custom model/threshold
    python test_deepfake_audio.py --audio audio.wav \
        --model best_classifier.pth \
        --threshold 0.45

Requirements:
    pip install torch torchaudio transformers soundfile numpy
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from pathlib import Path
from transformers import Wav2Vec2Model

# ─────────────────────────────────────────────
# 1.  Model definitions (must match training)
# ─────────────────────────────────────────────

class Classifier(nn.Module):
    """Lightweight MLP head trained on top of frozen Wav2Vec2 embeddings."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


class Wav2Vec2Classifier(nn.Module):
    """Full pipeline: Wav2Vec2 backbone + Classifier head."""
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        self.classifier = Classifier()

    def forward(self, waveform):
        outputs = self.wav2vec2(input_values=waveform)
        pooled  = outputs.last_hidden_state.mean(dim=1)   # mean-pool over time
        return self.classifier(pooled)


# ─────────────────────────────────────────────
# 2.  Audio pre-processing (same as training)
# ─────────────────────────────────────────────

SAMPLE_RATE = 16_000
DURATION    = 4.0
MAX_LENGTH  = int(SAMPLE_RATE * DURATION)   # 64 000 samples


def load_audio(path: str) -> torch.Tensor:
    """Load a .wav file and return a normalised (64000,) tensor."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform  = torch.from_numpy(data.T)            # (channels, samples)

    if waveform.shape[0] > 1:                       # stereo → mono
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)                  # (samples,)

    if sr != SAMPLE_RATE:                           # resample if needed
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    n = waveform.shape[0]
    if n < MAX_LENGTH:                              # pad or trim
        waveform = nn.functional.pad(waveform, (0, MAX_LENGTH - n))
    else:
        waveform = waveform[:MAX_LENGTH]

    # z-normalise (same as CustomDataset._load_audio)
    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)
    return waveform


# ─────────────────────────────────────────────
# 3.  Inference helpers
# ─────────────────────────────────────────────

def load_model(model_path: str, device: torch.device) -> Wav2Vec2Classifier:
    """
    Load the full Wav2Vec2Classifier.
    The checkpoint stores only BetterClassifier weights (fast_model), so
    we embed them inside the full pipeline automatically.
    """
    print(f"Loading Wav2Vec2 backbone from HuggingFace …")
    model = Wav2Vec2Classifier().to(device)

    print(f"Loading classifier weights from '{model_path}' …")
    state = torch.load(model_path, map_location=device)

    # Checkpoint may contain full-model keys or just classifier keys
    if any(k.startswith("wav2vec2.") for k in state):
        model.load_state_dict(state)
    else:
        # Only the Classifier head was saved (fast_model.state_dict())
        model.classifier.load_state_dict(state)

    model.eval()
    return model


@torch.no_grad()
def predict_file(model: Wav2Vec2Classifier,
                 path: str,
                 threshold: float,
                 device: torch.device) -> dict:
    """Return a prediction dict for one audio file."""
    waveform = load_audio(path).unsqueeze(0).to(device)   # (1, 64000)
    logits   = model(waveform)                            # (1, 2)
    prob_fake = torch.softmax(logits, dim=1)[0, 1].item()

    label = "FAKE" if prob_fake >= threshold else "REAL"
    return {
        "file":      os.path.basename(path),
        "prob_fake": prob_fake,
        "prob_real": 1 - prob_fake,
        "label":     label,
    }


def collect_wavs(path: str) -> list[str]:
    """Return a list of .wav file paths from a file or folder."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        wavs = sorted(p.glob("**/*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No .wav files found in '{path}'")
        return [str(w) for w in wavs]
    else:
        raise FileNotFoundError(f"Path not found: '{path}'")


# ─────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deepfake Audio Detector — Inference")
    parser.add_argument("--audio",     required=True,
                        help="Path to a .wav file OR a folder of .wav files")
    parser.add_argument("--model",     default="best_classifier.pth",
                        help="Path to the saved model weights (default: best_classifier.pth)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Fake-probability threshold (default: 0.5). "
                             "Use your EER threshold for best performance.")
    parser.add_argument("--device",    default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to run inference on (default: auto)")
    args = parser.parse_args()

    # ── Device ───────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")
    print(f"Threshold : {args.threshold:.4f}\n")

    # ── Load model ───────────────────────────
    model = load_model(args.model, device)

    # ── Collect files ────────────────────────
    wav_files = collect_wavs(args.audio)
    print(f"Found {len(wav_files)} file(s) to test.\n")
    print(f"{'File':<40} {'P(fake)':>8}  {'P(real)':>8}  {'Label':>6}")
    print("─" * 68)

    results = []
    for wav in wav_files:
        try:
            res = predict_file(model, wav, args.threshold, device)
            results.append(res)
            print(f"{res['file']:<40} {res['prob_fake']:>8.4f}  "
                  f"{res['prob_real']:>8.4f}  {res['label']:>6}")
        except Exception as e:
            print(f"{os.path.basename(wav):<40}  ERROR: {e}")

    # ── Summary ──────────────────────────────
    if results:
        n_fake = sum(1 for r in results if r["label"] == "FAKE")
        n_real = len(results) - n_fake
        print("─" * 68)
        print(f"Total: {len(results)}  |  REAL: {n_real}  |  FAKE: {n_fake}")


if __name__ == "__main__":
    main()
