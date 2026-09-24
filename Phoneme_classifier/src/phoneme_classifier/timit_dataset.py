from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .phone_maps import PAD_ID, PHONE_TO_ID_39, TIMIT_TO_39
from .utils import find_timit_utterances, flowhigh_mel_num_frames, load_audio_16k_mono


class TIMITFramePhonemeDataset(Dataset):
    """TIMIT frame-level phoneme dataset aligned to FlowHigh mel frames.

    Each item returns:
        waveform: [T]
        labels: [F_mel], where F_mel matches FlowHigh's mel frame count
        waveform_length: scalar number of samples
        label_length: scalar number of FlowHigh mel frames
    """

    def __init__(
        self,
        timit_root: str | Path,
        split: str = "TRAIN",
        max_samples: int | None = None,
        hop_length: int = 256,
        win_length: int = 1024,
        center: bool = True,
        drop_sa: bool = False,
    ) -> None:
        self.timit_root = Path(timit_root)
        self.split = split.upper()
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center

        self.items = find_timit_utterances(self.timit_root, self.split)

        # Optional: many TIMIT recipes remove SA1/SA2 because all speakers share them.
        if drop_sa:
            self.items = [
                (w, p)
                for (w, p) in self.items
                if not w.stem.upper().startswith("SA")
            ]

        if max_samples is not None:
            self.items = self.items[:max_samples]

    def __len__(self) -> int:
        return len(self.items)

    def _read_mapped_segments(self, phn_path: str | Path) -> list[tuple[int, int, str]]:
        """Read a TIMIT .PHN file and map raw phones to standard 39-phone labels.

        Raw TIMIT .PHN format:
            start_sample end_sample phone

        Example:
            0 3050 h#
            3050 5723 sh

        Standard mapping examples:
            h# -> sil
            ao -> aa
            zh -> sh
            q  -> None, discarded
        """
        segments: list[tuple[int, int, str]] = []

        with open(phn_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                start_s, end_s, phone = line.split()
                phone = phone.lower()

                if phone not in TIMIT_TO_39:
                    raise ValueError(f"Unknown TIMIT phone '{phone}' in {phn_path}")

                mapped = TIMIT_TO_39[phone]

                # In the standard TIMIT 39-phone mapping, q is discarded.
                if mapped is None:
                    continue

                segments.append((int(start_s), int(end_s), mapped))

        if not segments:
            raise RuntimeError(f"Empty PHN file after standard 39-phone mapping: {phn_path}")

        return segments

    def _segments_to_flowhigh_frame_labels(
        self,
        segments: list[tuple[int, int, str]],
        num_frames: int,
        num_samples: int,
    ) -> torch.Tensor:
        """Convert phone segments to labels on the FlowHigh mel-frame grid.

        Output:
            labels: [F_mel]

        If center=True:
            frame i corresponds approximately to sample i * hop_length.

        If center=False:
            frame i corresponds approximately to sample i * hop_length + win_length // 2.
        """
        labels = torch.empty(num_frames, dtype=torch.long)
        seg_idx = 0

        for frame_idx in range(num_frames):
            if self.center:
                # For centered STFT, frame i is centered at i * hop_length in the
                # original signal because the signal is padded by n_fft/2.
                center_sample = frame_idx * self.hop_length
            else:
                # For non-centered STFT, the center is i * hop + win_length / 2.
                center_sample = frame_idx * self.hop_length + self.win_length // 2

            # Clamp to valid audio region. This matters for edge frames.
            center_sample = min(max(center_sample, 0), max(num_samples - 1, 0))

            while seg_idx < len(segments) - 1 and center_sample >= segments[seg_idx][1]:
                seg_idx += 1

            phone = segments[seg_idx][2]
            labels[frame_idx] = PHONE_TO_ID_39[phone]

        return labels

    def __getitem__(self, idx: int) -> dict[str, Any]:
        wav_path, phn_path = self.items[idx]

        waveform = load_audio_16k_mono(wav_path)  # [T]
        num_samples = int(waveform.numel())

        num_frames = flowhigh_mel_num_frames(
            num_samples,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
        )

        segments = self._read_mapped_segments(phn_path)
        labels = self._segments_to_flowhigh_frame_labels(
            segments=segments,
            num_frames=num_frames,
            num_samples=num_samples,
        )

        return {
            "waveform": waveform,
            "labels": labels,
            "waveform_length": num_samples,
            "label_length": num_frames,
            "wav_path": str(wav_path),
            "phn_path": str(phn_path),
        }


def collate_timit_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length TIMIT examples into a batch.

    Waveforms are padded with zeros.
    Labels are padded with PAD_ID, which is ignored by CrossEntropyLoss.
    """
    max_wav_len = max(x["waveform"].numel() for x in batch)
    max_label_len = max(x["labels"].numel() for x in batch)

    waveforms = torch.zeros(len(batch), max_wav_len, dtype=torch.float32)
    labels = torch.full((len(batch), max_label_len), PAD_ID, dtype=torch.long)

    waveform_lengths = torch.empty(len(batch), dtype=torch.long)
    label_lengths = torch.empty(len(batch), dtype=torch.long)

    wav_paths = []
    phn_paths = []

    for i, item in enumerate(batch):
        wav = item["waveform"]
        lab = item["labels"]

        waveforms[i, : wav.numel()] = wav
        labels[i, : lab.numel()] = lab

        waveform_lengths[i] = item["waveform_length"]
        label_lengths[i] = item["label_length"]

        wav_paths.append(item["wav_path"])
        phn_paths.append(item["phn_path"])

    return {
        "waveforms": waveforms,
        "waveform_lengths": waveform_lengths,
        "labels": labels,
        "label_lengths": label_lengths,
        "wav_paths": wav_paths,
        "phn_paths": phn_paths,
    }
