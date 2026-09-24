from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from scipy.signal import cheby1, resample_poly, sosfiltfilt

from src.phoneme_classifier.model import HubertFlowHighPhonemeClassifier
from src.phoneme_classifier.phone_maps import (
    ID_TO_PHONE_39,
    PAD_ID,
    PHONE_TO_ID_39,
)
from src.phoneme_classifier.timit_dataset import TIMITFramePhonemeDataset
from src.phoneme_classifier.utils import flowhigh_mel_num_frames, load_audio_16k_mono


SOURCE_SR = 16000
FILTER_ORDER = 8
FILTER_RIPPLE = 0.05


def str_to_bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"true", "1", "yes", "y", "t"}


def find_phn_path(wav_path: Path) -> Path:
    """Robustly find the corresponding TIMIT .PHN path."""
    candidates = [
        wav_path.with_suffix(".PHN"),
        wav_path.with_suffix(".phn"),
    ]

    name = wav_path.name

    # Handle names like SA1.WAV.wav -> SA1.PHN
    if name.lower().endswith(".wav.wav"):
        base = name[:-8]
        candidates.append(wav_path.with_name(base + ".PHN"))
        candidates.append(wav_path.with_name(base + ".phn"))

    # Handle names like SA1.WAV -> SA1.PHN or SA1.wav -> SA1.PHN
    if name.lower().endswith(".wav"):
        base = name[:-4]
        candidates.append(wav_path.with_name(base + ".PHN"))
        candidates.append(wav_path.with_name(base + ".phn"))

    for cand in candidates:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"Could not find PHN file for {wav_path}. Tried:\n"
        + "\n".join(str(c) for c in candidates)
    )


def rms_normalize_waveform(
    waveform: torch.Tensor,
    target_rms: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """RMS-normalize a single waveform.

    Args:
        waveform: [T]
        target_rms: target RMS value.
        eps: numerical stability.

    Returns:
        RMS-normalized waveform [T].
    """
    out = waveform.clone()
    rms = torch.sqrt(torch.mean(out ** 2) + eps)

    if rms > eps:
        out = out * (target_rms / rms)

    return out


def downsample_then_upsample_keep_16k(
    waveform: torch.Tensor,
    downsample_sr: int,
    source_sr: int = SOURCE_SR,
    filter_order: int = FILTER_ORDER,
    filter_ripple: float = FILTER_RIPPLE,
) -> torch.Tensor:
    """Low-pass, downsample, then upsample back to 16 kHz.

    Input:
        waveform: [T], 16 kHz

    Output:
        waveform_bl: [T], still 16 kHz, but band-limited.

    No peak normalization is applied here.
    RMS normalization is applied later if enabled.
    """
    if downsample_sr <= 0:
        raise ValueError(f"downsample_sr must be positive, got {downsample_sr}")

    if downsample_sr >= source_sr:
        raise ValueError(
            f"downsample_sr must be lower than source_sr, got "
            f"{downsample_sr} >= {source_sr}"
        )

    wav = waveform.detach().cpu().float().numpy()
    length = wav.shape[0]

    nyq = source_sr / 2.0
    cutoff = downsample_sr / 2.0
    normalized_cutoff = cutoff / nyq

    if not (0.0 < normalized_cutoff < 1.0):
        raise ValueError(
            f"Invalid normalized cutoff: source_sr={source_sr}, "
            f"downsample_sr={downsample_sr}, normalized_cutoff={normalized_cutoff}"
        )

    sos = cheby1(
        filter_order,
        filter_ripple,
        normalized_cutoff,
        btype="lowpass",
        output="sos",
    )

    filtered = sosfiltfilt(sos, wav)
    down = resample_poly(filtered, downsample_sr, source_sr)
    up = resample_poly(down, source_sr, downsample_sr)

    if up.shape[0] > length:
        up = up[:length]
    elif up.shape[0] < length:
        up = np.pad(up, (0, length - up.shape[0]))

    return torch.from_numpy(up.astype(np.float32))


def load_model(
    ckpt_path: str | Path,
    device: torch.device,
) -> tuple[HubertFlowHighPhonemeClassifier, dict[int, str], dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    phones = ckpt.get("phones", None)
    if phones is not None:
        id_to_phone = {i: p for i, p in enumerate(phones)}
        num_phonemes = len(phones)
    else:
        id_to_phone = ID_TO_PHONE_39
        num_phonemes = len(PHONE_TO_ID_39)

    model = HubertFlowHighPhonemeClassifier(
        num_phonemes=num_phonemes,
        freeze_hubert=False,
    )

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        raise KeyError(f"Unknown checkpoint format. Keys: {list(ckpt.keys())}")

    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    return model, id_to_phone, ckpt


def resolve_rms_setting(
    args: argparse.Namespace,
    ckpt: dict,
) -> tuple[bool, float]:
    """Use command-line RMS setting if provided; otherwise use checkpoint setting."""
    ckpt_rms_normalize = bool(ckpt.get("rms_normalize", False))
    ckpt_target_rms = float(ckpt.get("target_rms", 0.05))

    if args.rms_normalize is None:
        rms_normalize = ckpt_rms_normalize
    else:
        rms_normalize = args.rms_normalize

    if args.target_rms is None:
        target_rms = ckpt_target_rms
    else:
        target_rms = args.target_rms

    return rms_normalize, target_rms


def get_ground_truth_labels(
    wav_path: Path,
    phn_path: Path,
    num_frames: int,
    num_samples: int,
    center: bool = True,
    hop_length: int = 256,
    win_length: int = 1024,
) -> torch.Tensor:
    """Reuse the dataset's PHN parsing and frame-label conversion."""
    # For /scratch/ningyuan/TIMIT/data/TEST/DR1/FAKS0/SA1.WAV.wav,
    # parents[3] should be /scratch/ningyuan/TIMIT/data.
    dummy_root = wav_path.parents[3] if len(wav_path.parents) >= 4 else wav_path.parent

    dataset = TIMITFramePhonemeDataset(
        timit_root=dummy_root,
        split="TEST",
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        drop_sa=False,
    )

    segments = dataset._read_mapped_segments(phn_path)
    labels = dataset._segments_to_flowhigh_frame_labels(
        segments=segments,
        num_frames=num_frames,
        num_samples=num_samples,
    )

    return labels


def plot_waveform_and_labels(
    waveform: torch.Tensor,
    gt_ids: torch.Tensor,
    pred_ids: torch.Tensor,
    id_to_phone: dict[int, str],
    save_path: str | Path,
    sample_rate: int = SOURCE_SR,
    hop_length: int = 256,
    title: str = "",
    zoom_start: float | None = None,
    zoom_end: float | None = None,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    axis_label_fontsize = 18
    tick_label_fontsize = 16
    title_fontsize = 18
    legend_fontsize = 14

    waveform_np = waveform.detach().cpu().numpy()
    num_samples = waveform_np.shape[0]
    wav_times = np.arange(num_samples) / sample_rate

    gt_ids_np = gt_ids.detach().cpu().numpy()
    pred_ids_np = pred_ids.detach().cpu().numpy()

    num_frames = len(gt_ids_np)
    frame_times = np.arange(num_frames) * hop_length / sample_rate

    valid = gt_ids_np != PAD_ID
    gt_ids_np = gt_ids_np[valid]
    pred_ids_np = pred_ids_np[valid]
    frame_times = frame_times[valid]

    if zoom_start is not None:
        keep = frame_times >= zoom_start
        gt_ids_np = gt_ids_np[keep]
        pred_ids_np = pred_ids_np[keep]
        frame_times = frame_times[keep]

    if zoom_end is not None:
        keep = frame_times <= zoom_end
        gt_ids_np = gt_ids_np[keep]
        pred_ids_np = pred_ids_np[keep]
        frame_times = frame_times[keep]

    # Only show labels used in this utterance / selected region:
    # all ground-truth labels plus all wrong predicted labels.
    wrong_mask = pred_ids_np != gt_ids_np
    used_ids = sorted(set(gt_ids_np.tolist()) | set(pred_ids_np[wrong_mask].tolist()))

    id_to_y = {pid: i for i, pid in enumerate(used_ids)}
    y_labels = [id_to_phone[int(pid)] for pid in used_ids]

    gt_y = np.array([id_to_y[int(x)] for x in gt_ids_np])
    pred_y = np.array([id_to_y[int(x)] for x in pred_ids_np])

    correct_mask = pred_ids_np == gt_ids_np
    wrong_mask = ~correct_mask

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(20, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.6]},
    )

    ax_wav, ax_lab = axes

    ax_wav.plot(wav_times, waveform_np, linewidth=0.8)
    ax_wav.set_ylabel("Amplitude", fontsize=axis_label_fontsize)
    ax_wav.set_title(title, fontsize=title_fontsize)
    ax_wav.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax_wav.grid(True, alpha=0.25)

    # Ground truth: blue square markers.
    ax_lab.scatter(
        frame_times,
        gt_y,
        s=16,
        c="blue",
        marker="s",
        label="Ground truth",
        alpha=0.75,
        zorder=3,
    )

    # Correct predictions: green circles, slightly shifted upward.
    ax_lab.scatter(
        frame_times[correct_mask],
        pred_y[correct_mask] + 0.18,
        s=14,
        c="green",
        marker="o",
        label="Prediction correct",
        alpha=0.85,
        zorder=4,
    )

    # Wrong predictions: red x markers, slightly shifted upward.
    ax_lab.scatter(
        frame_times[wrong_mask],
        pred_y[wrong_mask] + 0.18,
        s=24,
        c="red",
        marker="x",
        label="Prediction wrong",
        alpha=0.95,
        zorder=5,
    )

    ax_lab.set_yticks(range(len(y_labels)))
    ax_lab.set_yticklabels(y_labels, fontsize=tick_label_fontsize)

    # Dashed horizontal guide lines for each phoneme row.
    for y in range(len(y_labels)):
        ax_lab.axhline(
            y=y,
            linestyle="--",
            linewidth=0.7,
            alpha=0.35,
            zorder=0,
        )

    ax_lab.set_xlabel("Time (s)", fontsize=axis_label_fontsize)
    ax_lab.set_ylabel("Used phoneme labels", fontsize=axis_label_fontsize)
    ax_lab.tick_params(axis="x", labelsize=tick_label_fontsize)
    ax_lab.tick_params(axis="y", labelsize=tick_label_fontsize)

    # Vertical grid lines for time.
    ax_lab.grid(True, axis="x", alpha=0.25)
    ax_lab.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=legend_fontsize,
        frameon=True,
    )

    if zoom_start is not None or zoom_end is not None:
        left = zoom_start if zoom_start is not None else wav_times[0]
        right = zoom_end if zoom_end is not None else wav_times[-1]
        ax_wav.set_xlim(left, right)
        ax_lab.set_xlim(left, right)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize waveform, ground-truth phones, and predicted phones."
    )

    parser.add_argument(
        "--wav",
        type=str,
        default="/scratch/ningyuan/TIMIT/data/TEST/DR1/FAKS0/SA1.WAV.wav",
        help="Path to one TIMIT wav file.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained checkpoint, e.g., exp2_bl_2k/best.pt.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/scratch/ningyuan/Phoneme_classifier/transition_region.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--downsample_sr",
        type=int,
        default=2000,
        help="Intermediate low sampling rate. 2000 means cutoff = 1000 Hz.",
    )
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--center", type=str_to_bool, default=True)

    parser.add_argument(
        "--rms_normalize",
        type=str_to_bool,
        default=None,
        help=(
            "Whether to RMS-normalize after degradation. "
            "Default: use checkpoint setting if available."
        ),
    )
    parser.add_argument(
        "--target_rms",
        type=float,
        default=None,
        help=(
            "Target RMS. Default: use checkpoint setting if available, "
            "otherwise 0.05."
        ),
    )

    parser.add_argument("--zoom_start", type=float, default=None)
    parser.add_argument("--zoom_end", type=float, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    wav_path = Path(args.wav)
    phn_path = find_phn_path(wav_path)
    device = torch.device(args.device)

    waveform = load_audio_16k_mono(wav_path)  # [T], 16 kHz
    num_samples = int(waveform.numel())

    num_frames = flowhigh_mel_num_frames(
        num_samples,
        hop_length=args.hop_length,
        win_length=args.win_length,
        center=args.center,
    )

    gt_ids = get_ground_truth_labels(
        wav_path=wav_path,
        phn_path=phn_path,
        num_frames=num_frames,
        num_samples=num_samples,
        center=args.center,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )

    waveform_bl = downsample_then_upsample_keep_16k(
        waveform=waveform,
        downsample_sr=args.downsample_sr,
        source_sr=SOURCE_SR,
    )

    model, id_to_phone, ckpt = load_model(args.ckpt, device=device)

    rms_normalize, target_rms = resolve_rms_setting(args, ckpt)

    if rms_normalize:
        waveform_bl = rms_normalize_waveform(
            waveform=waveform_bl,
            target_rms=target_rms,
        )

    with torch.no_grad():
        waveforms = waveform_bl.unsqueeze(0).to(device)  # [1, T]
        waveform_lengths = torch.tensor([num_samples], dtype=torch.long, device=device)

        logits, _ = model(
            waveforms=waveforms,
            waveform_lengths=waveform_lengths,
            target_num_frames=num_frames,
        )

        pred_ids = logits.argmax(dim=-1).squeeze(0).cpu()  # [F_mel]

    frame_acc = (pred_ids == gt_ids).float().mean().item()

    title = (
        f"{wav_path.name} | downsample_sr={args.downsample_sr}, "
        f"cutoff={args.downsample_sr / 2:.0f} Hz | "
        f"RMS={rms_normalize}, target={target_rms} | "
        f"frame acc={frame_acc:.3f}"
    )

    plot_waveform_and_labels(
        waveform=waveform_bl,
        gt_ids=gt_ids,
        pred_ids=pred_ids,
        id_to_phone=id_to_phone,
        save_path=args.out,
        sample_rate=SOURCE_SR,
        hop_length=args.hop_length,
        title=title,
        zoom_start=args.zoom_start,
        zoom_end=args.zoom_end,
    )

    print(f"WAV: {wav_path}")
    print(f"PHN: {phn_path}")
    print(f"Frames: {num_frames}")
    print("Peak normalization before degradation: False")
    print(f"RMS normalization after degradation: {rms_normalize}")
    print(f"Target RMS: {target_rms}")
    print(f"Frame accuracy: {frame_acc:.4f}")
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()