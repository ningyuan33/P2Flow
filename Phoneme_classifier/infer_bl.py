from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import cheby1, resample_poly, sosfiltfilt
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.phoneme_classifier.model import HubertFlowHighPhonemeClassifier
from src.phoneme_classifier.phone_maps import ID_TO_PHONE_39, PAD_ID, PHONE_TO_ID_39
from src.phoneme_classifier.timit_dataset import (
    TIMITFramePhonemeDataset,
    collate_timit_batch,
)


SOURCE_SR = 16000
FILTER_ORDER = 8
FILTER_RIPPLE = 0.05


def str2bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x

    x = x.lower()
    if x in {"true", "1", "yes", "y", "t"}:
        return True
    if x in {"false", "0", "no", "n", "f"}:
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def rms_normalize_batch(
    waveforms: torch.Tensor,
    waveform_lengths: torch.Tensor,
    target_rms: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Length-aware RMS normalization for padded waveform batches.

    Args:
        waveforms: [B, T_max]
        waveform_lengths: [B]
        target_rms: target RMS value
        eps: numerical stability

    Returns:
        RMS-normalized waveforms with padding unchanged.
    """
    out = waveforms.clone()

    for i in range(out.shape[0]):
        length = int(waveform_lengths[i].item())

        if length <= 0:
            continue

        wav = out[i, :length]
        rms = torch.sqrt(torch.mean(wav ** 2) + eps)

        if rms > eps:
            out[i, :length] = wav * (target_rms / rms)

    return out


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


def downsample_then_upsample_keep_16k(
    waveforms: torch.Tensor,
    waveform_lengths: torch.Tensor,
    downsample_sr: int,
    source_sr: int = SOURCE_SR,
    filter_order: int = FILTER_ORDER,
    filter_ripple: float = FILTER_RIPPLE,
) -> torch.Tensor:
    """Create low-SR degraded waveforms, then return them at 16 kHz.

    Pipeline:
        16 kHz waveform
            -> Chebyshev low-pass filter, cutoff = downsample_sr / 2
            -> downsample to downsample_sr
            -> upsample back to 16 kHz

    No peak normalization is applied here. If enabled, RMS normalization is
    applied after degradation and before HuBERT.
    """
    if downsample_sr <= 0:
        raise ValueError(f"downsample_sr must be positive, got {downsample_sr}")

    if downsample_sr >= source_sr:
        raise ValueError(
            f"downsample_sr must be lower than source_sr for degradation, "
            f"got downsample_sr={downsample_sr}, source_sr={source_sr}"
        )

    device = waveforms.device
    dtype = waveforms.dtype

    wave_np = waveforms.detach().cpu().float().numpy()
    lengths_np = waveform_lengths.detach().cpu().long().numpy()

    out = np.zeros_like(wave_np, dtype=np.float32)

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

    for i in range(wave_np.shape[0]):
        length = int(lengths_np[i])
        wav = wave_np[i, :length]

        if wav.size == 0:
            continue

        # 1. Low-pass filter.
        filtered = sosfiltfilt(sos, wav)

        # 2. Downsample to low SR.
        down = resample_poly(filtered, downsample_sr, source_sr)

        # 3. Upsample back to 16 kHz.
        up = resample_poly(down, source_sr, downsample_sr)

        # 4. Match original true length exactly.
        if up.shape[0] > length:
            up = up[:length]
        elif up.shape[0] < length:
            up = np.pad(up, (0, length - up.shape[0]))

        out[i, :length] = up.astype(np.float32)

    return torch.from_numpy(out).to(device=device, dtype=dtype)


def load_model_from_checkpoint(
    ckpt_path: str | Path,
    device: torch.device,
) -> tuple[HubertFlowHighPhonemeClassifier, dict[int, str], dict]:
    ckpt_path = Path(ckpt_path)
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
        raise KeyError(
            f"Checkpoint {ckpt_path} does not contain 'model_state_dict' or 'model'. "
            f"Available keys: {list(ckpt.keys())}"
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    return model, id_to_phone, ckpt


def update_confusion_matrix(
    confusion: torch.Tensor,
    pred: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = PAD_ID,
) -> None:
    """Update frame-level confusion matrix.

    Rows:
        ground-truth labels
    Columns:
        predicted labels
    """
    mask = labels != ignore_index

    gold_flat = labels[mask].view(-1).cpu()
    pred_flat = pred[mask].view(-1).cpu()

    indices = gold_flat * num_classes + pred_flat
    counts = torch.bincount(indices, minlength=num_classes * num_classes)

    confusion += counts.reshape(num_classes, num_classes)


def save_confusion_matrix(
    confusion: torch.Tensor,
    id_to_phone: dict[int, str],
    save_path: str | Path,
    normalize: bool = True,
    title_suffix: str = "",
) -> None:
    """Save phoneme confusion matrix figure.

    Rows:
        ground-truth phonemes
    Columns:
        predicted phonemes
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion.float()

    if normalize:
        row_sums = cm.sum(dim=1, keepdim=True).clamp_min(1.0)
        cm_plot = cm / row_sums
        colorbar_label = "Row-normalized proportion"
        title = "Band-Limited Frame-Level Phoneme Confusion Matrix"
    else:
        cm_plot = cm
        colorbar_label = "Count"
        title = "Band-Limited Frame-Level Phoneme Confusion Matrix"

    if title_suffix:
        title = f"{title} ({title_suffix})"

    num_classes = cm.shape[0]
    labels = [id_to_phone[i] for i in range(num_classes)]

    plt.figure(figsize=(14, 12))
    im = plt.imshow(cm_plot.numpy(), aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04, label=colorbar_label)

    plt.xticks(range(num_classes), labels, rotation=90, fontsize=8)
    plt.yticks(range(num_classes), labels, fontsize=8)

    plt.xlabel("Predicted phoneme")
    plt.ylabel("Ground-truth phoneme")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()


def save_confusion_matrix_csv(
    confusion: torch.Tensor,
    id_to_phone: dict[int, str],
    save_path: str | Path,
) -> None:
    """Save raw confusion matrix counts as CSV."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = confusion.shape[0]
    labels = [id_to_phone[i] for i in range(num_classes)]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gold/pred"] + labels)

        for i in range(num_classes):
            writer.writerow([labels[i]] + confusion[i].tolist())


def evaluate_bandlimited(
    model: HubertFlowHighPhonemeClassifier,
    loader: DataLoader,
    id_to_phone: dict[int, str],
    device: torch.device,
    downsample_sr: int,
    rms_normalize: bool,
    target_rms: float,
) -> tuple[dict, list[dict], torch.Tensor]:
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    rows: list[dict] = []

    num_classes = len(id_to_phone)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"TIMIT TEST downsample_sr={downsample_sr}"):
            waveforms = batch["waveforms"].to(device)
            waveform_lengths = batch["waveform_lengths"].to(device)
            labels = batch["labels"].to(device)

            # Create low-SR degraded waveform, but keep HuBERT input at 16 kHz.
            waveforms_bl = downsample_then_upsample_keep_16k(
                waveforms=waveforms,
                waveform_lengths=waveform_lengths,
                downsample_sr=downsample_sr,
                source_sr=SOURCE_SR,
                filter_order=FILTER_ORDER,
                filter_ripple=FILTER_RIPPLE,
            )

            # Apply RMS normalization after degradation, matching train_bl.py.
            if rms_normalize:
                waveforms_bl = rms_normalize_batch(
                    waveforms=waveforms_bl,
                    waveform_lengths=waveform_lengths,
                    target_rms=target_rms,
                )

            target_num_frames = labels.shape[1]

            logits, _ = model(
                waveforms=waveforms_bl,
                waveform_lengths=waveform_lengths,
                target_num_frames=target_num_frames,
            )

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=PAD_ID,
            )

            pred = logits.argmax(dim=-1)
            mask = labels != PAD_ID

            correct = ((pred == labels) & mask).sum().item()
            count = mask.sum().item()

            total_loss += loss.item() * max(count, 1)
            total_correct += correct
            total_count += count

            update_confusion_matrix(
                confusion=confusion,
                pred=pred,
                labels=labels,
                num_classes=num_classes,
                ignore_index=PAD_ID,
            )

            pred_cpu = pred.cpu()
            labels_cpu = labels.cpu()

            for i, wav_path in enumerate(batch["wav_paths"]):
                valid_len = int(batch["label_lengths"][i].item())

                pred_ids = pred_cpu[i, :valid_len].tolist()
                gold_ids = labels_cpu[i, :valid_len].tolist()

                pred_phones = [id_to_phone[int(x)] for x in pred_ids]
                gold_phones = [id_to_phone[int(x)] for x in gold_ids]

                utt_correct = sum(int(p == g) for p, g in zip(pred_ids, gold_ids))
                utt_total = len(gold_ids)
                utt_acc = utt_correct / max(utt_total, 1)

                rows.append(
                    {
                        "wav_path": wav_path,
                        "phn_path": batch["phn_paths"][i],
                        "downsample_sr": downsample_sr,
                        "lowpass_cutoff_hz": downsample_sr / 2.0,
                        "num_frames": valid_len,
                        "frame_accuracy": utt_acc,
                        "pred_ids": " ".join(map(str, pred_ids)),
                        "gold_ids": " ".join(map(str, gold_ids)),
                        "pred_phones": " ".join(pred_phones),
                        "gold_phones": " ".join(gold_phones),
                    }
                )

    test_loss = total_loss / max(total_count, 1)
    test_acc = total_correct / max(total_count, 1)

    per_phone_total = confusion.sum(dim=1)
    per_phone_correct = confusion.diag()
    per_phone_acc = per_phone_correct.float() / per_phone_total.clamp_min(1).float()

    metrics = {
        "split": "TEST",
        "usage": "bandlimited_final_inference_or_evaluation_only",
        "source_sample_rate": SOURCE_SR,
        "input_to_model_sample_rate": SOURCE_SR,
        "downsample_sr": downsample_sr,
        "lowpass_cutoff_hz": downsample_sr / 2.0,
        "filter_order": FILTER_ORDER,
        "filter_ripple": FILTER_RIPPLE,
        "peak_normalize_before_degradation": False,
        "rms_normalize_after_degradation": rms_normalize,
        "target_rms": target_rms,
        "test_loss": test_loss,
        "test_frame_accuracy": test_acc,
        "num_scored_frames": total_count,
        "per_phone_accuracy": {
            id_to_phone[i]: float(per_phone_acc[i].item())
            for i in range(num_classes)
        },
        "per_phone_total_frames": {
            id_to_phone[i]: int(per_phone_total[i].item())
            for i in range(num_classes)
        },
    }

    return metrics, rows, confusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run band-limited inference/evaluation on official TIMIT TEST split. "
            "The input is low-pass filtered, downsampled to --downsample_sr, "
            "then upsampled back to 16 kHz before HuBERT."
        )
    )

    parser.add_argument(
        "--timit_root",
        type=str,
        required=True,
        help="TIMIT root",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained checkpoint, usually best.pt.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for metrics, predictions, and confusion matrix.",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--downsample_sr",
        type=int,
        default=1200,
        help=(
            "Intermediate low sampling rate. The cutoff is downsample_sr / 2. "
            "The waveform is then upsampled back to 16 kHz for HuBERT. "
            "Examples: 1200, 2000, 4000, 8000."
        ),
    )

    parser.add_argument(
        "--rms_normalize",
        type=str2bool,
        default=None,
        help=(
            "Whether to apply RMS normalization after degradation. "
            "Default: use checkpoint setting if available."
        ),
    )
    parser.add_argument(
        "--target_rms",
        type=float,
        default=None,
        help=(
            "Target RMS value. Default: use checkpoint setting if available, "
            "otherwise 0.05."
        ),
    )

    parser.add_argument(
        "--center",
        type=str2bool,
        default=True,
        help="Whether FlowHigh mel STFT uses center=True.",
    )

    parser.add_argument(
        "--drop_sa",
        type=str2bool,
        default=False,
        help="Whether to drop SA1/SA2 from TEST. Default false.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--no_normalized_cm",
        action="store_true",
        help="If set, save raw-count confusion matrix figure instead of row-normalized figure.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, id_to_phone, ckpt = load_model_from_checkpoint(
        ckpt_path=args.ckpt,
        device=device,
    )

    # Prefer checkpoint setting, so inference matches train_bl.py.
    ckpt_rms = bool(ckpt.get("rms_normalize", False))
    ckpt_target_rms = float(ckpt.get("target_rms", 0.05))

    rms_normalize = ckpt_rms if args.rms_normalize is None else args.rms_normalize
    target_rms = ckpt_target_rms if args.target_rms is None else args.target_rms

    print("Band-limited inference preprocessing:")
    print("Peak normalization before degradation: False")
    print(f"RMS normalization after degradation: {rms_normalize}")
    print(f"Target RMS: {target_rms}")
    print(f"Downsample SR: {args.downsample_sr}")
    print(f"Low-pass cutoff: {args.downsample_sr / 2.0} Hz")

    test_set = TIMITFramePhonemeDataset(
        timit_root=args.timit_root,
        split="TEST",
        center=args.center,
        drop_sa=args.drop_sa,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_timit_batch,
    )

    metrics, rows, confusion = evaluate_bandlimited(
        model=model,
        loader=test_loader,
        id_to_phone=id_to_phone,
        device=device,
        downsample_sr=args.downsample_sr,
        rms_normalize=rms_normalize,
        target_rms=target_rms,
    )

    metrics.update(
        {
            "checkpoint": str(args.ckpt),
            "drop_sa": args.drop_sa,
            "num_utterances": len(test_set),
            "num_phoneme_labels": len(id_to_phone),
            "best_valid_acc_from_ckpt": ckpt.get("best_valid_acc", None),
            "best_valid_loss_from_ckpt": ckpt.get("best_valid_loss", None),
            "epoch_from_ckpt": ckpt.get("epoch", None),
        }
    )

    metrics_path = out_dir / "test_metrics.json"
    predictions_path = out_dir / "test_predictions.csv"
    confusion_csv_path = out_dir / "confusion_matrix_counts.csv"
    confusion_png_path = out_dir / "confusion_matrix.png"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "wav_path",
                "phn_path",
                "downsample_sr",
                "lowpass_cutoff_hz",
                "num_frames",
                "frame_accuracy",
                "pred_ids",
                "gold_ids",
                "pred_phones",
                "gold_phones",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    save_confusion_matrix_csv(
        confusion=confusion,
        id_to_phone=id_to_phone,
        save_path=confusion_csv_path,
    )

    save_confusion_matrix(
        confusion=confusion,
        id_to_phone=id_to_phone,
        save_path=confusion_png_path,
        normalize=not args.no_normalized_cm,
        title_suffix=(
            f"downsample_sr={args.downsample_sr}, "
            f"cutoff={args.downsample_sr / 2.0:.0f} Hz"
        ),
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved confusion matrix counts to: {confusion_csv_path}")
    print(f"Saved confusion matrix figure to: {confusion_png_path}")


if __name__ == "__main__":
    main()