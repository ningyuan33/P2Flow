from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.pipelines import HUBERT_BASE


class HubertFlowHighPhonemeClassifier(nn.Module):
    """HuBERT Base frame-level phoneme classifier aligned to FlowHigh frames.

    Input:
        waveforms: [B, T]
        waveform_lengths: optional [B]
        target_num_frames: FlowHigh mel frame length after batch padding

    Output:
        logits: [B, target_num_frames, num_phonemes]
    """

    def __init__(
        self,
        num_phonemes: int = 39,
        freeze_hubert: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hubert = HUBERT_BASE.get_model()
        self.hidden_size = HUBERT_BASE._params["encoder_embed_dim"]  # 768 for HuBERT Base
        self.num_phonemes = num_phonemes
        self.freeze_hubert = freeze_hubert

        self.dropout = nn.Dropout(dropout)
        self.phoneme_head = nn.Linear(self.hidden_size, num_phonemes)

        # Initialize only the newly added classifier head.
        self._init_phoneme_head()

        if freeze_hubert:
            for p in self.hubert.parameters():
                p.requires_grad = False

    def _init_phoneme_head(self) -> None:
        """Apply Kaiming initialization to the phoneme classifier head only."""
        nn.init.kaiming_normal_(
            self.phoneme_head.weight,
            mode="fan_out",
            nonlinearity="linear",
        )

        if self.phoneme_head.bias is not None:
            nn.init.zeros_(self.phoneme_head.bias)

    def extract_hubert(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.freeze_hubert:
            with torch.no_grad():
                features, feature_lengths = self.hubert(waveforms, waveform_lengths)
        else:
            features, feature_lengths = self.hubert(waveforms, waveform_lengths)

        return features, feature_lengths

    @staticmethod
    def align_to_flowhigh_frames(
        features: torch.Tensor,
        target_num_frames: int,
    ) -> torch.Tensor:
        """Linearly interpolate [B, F_hubert, H] to [B, F_mel, H]."""
        if features.size(1) == target_num_frames:
            return features

        return F.interpolate(
            features.transpose(1, 2),  # [B, H, F_hubert]
            size=target_num_frames,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)  # [B, F_mel, H]

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor | None = None,
        target_num_frames: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features, feature_lengths = self.extract_hubert(waveforms, waveform_lengths)
        # features: [B, F_hubert, 768]

        if target_num_frames is not None:
            features = self.align_to_flowhigh_frames(features, target_num_frames)
            # After interpolation, native HuBERT feature_lengths are no longer valid.
            feature_lengths = None

        logits = self.phoneme_head(self.dropout(features))
        # logits: [B, F_mel, 39]

        return logits, feature_lengths

    @torch.no_grad()
    def predict_proba(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor | None = None,
        target_num_frames: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        logits, _ = self.forward(waveforms, waveform_lengths, target_num_frames)
        return torch.softmax(logits, dim=-1)
