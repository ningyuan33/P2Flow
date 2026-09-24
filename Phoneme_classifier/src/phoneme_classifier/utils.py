from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torchaudio


def find_timit_utterances(timit_root: str | Path, split: str) -> list[tuple[Path, Path]]:
    """Return sorted `(wav_path, phn_path)` pairs for a TIMIT split.

    `split` should be `TRAIN` or `TEST`. The function supports uppercase/lowercase
    file extensions.
    """
    root = Path(timit_root)
    split_dir = root / split.upper()
    if not split_dir.exists():
        split_dir = root / split.lower()
    if not split_dir.exists():
        raise FileNotFoundError(f"Could not find split directory: {root}/{split}")

    wavs = []
    for pat in ("*.WAV", "*.wav"):
        wavs.extend(split_dir.rglob(pat))

    pairs = []
    for wav in sorted(wavs):
        candidates = [wav.with_suffix(".PHN"), wav.with_suffix(".phn")]
        phn = next((p for p in candidates if p.exists()), None)
        if phn is not None:
            pairs.append((wav, phn))

    if not pairs:
        raise RuntimeError(f"No WAV/PHN pairs found under {split_dir}")
    return pairs


def load_audio_16k_mono(path: str | Path) -> torch.Tensor:
    """Load waveform as mono float tensor `[T]` at 16 kHz."""
    wav, sr = torchaudio.load(str(path))
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0).contiguous()


def flowhigh_mel_num_frames(
    num_samples: int,
    hop_length: int = 256,
    win_length: int = 1024,
    center: bool = True,
) -> int:
    """Compute the expected number of FlowHigh mel frames.

    FlowHigh uses hop_length=256 at 16 kHz, i.e. 16 ms per frame. Most
    PyTorch/SpeechBrain STFT pipelines use centered framing. With centered STFT,
    the frame count is `1 + floor(num_samples / hop_length)`.

    If your local FlowHigh `mel_spectogram` returns a different value, change
    `center` or directly replace this function with a call to the actual encoder
    and use `mel.shape[1]`.
    """
    if center:
        return int(num_samples // hop_length + 1)
    if num_samples < win_length:
        return 1
    return int((num_samples - win_length) // hop_length + 1)


def make_length_tensor(lengths: Iterable[int], device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(list(lengths), dtype=torch.long, device=device)
