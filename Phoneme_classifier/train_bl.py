from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import cheby1, resample_poly, sosfiltfilt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.phoneme_classifier.model import HubertFlowHighPhonemeClassifier
from src.phoneme_classifier.phone_maps import PAD_ID, PHONES_39
from src.phoneme_classifier.timit_dataset import (
    TIMITFramePhonemeDataset,
    collate_timit_batch,
)


SOURCE_SR = 16000
FILTER_ORDER = 8
FILTER_RIPPLE = 0.05


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v

    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        target_rms: target RMS value for each utterance
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

    No peak normalization is applied.
    RMS normalization is applied later before HuBERT if enabled.

    Args:
        waveforms:
            Padded waveform tensor, [B, T_max], sampled at 16 kHz.
        waveform_lengths:
            True waveform lengths before padding, [B].
        downsample_sr:
            Intermediate low sampling rate, e.g., 1200, 2000, 4000.
        source_sr:
            Original and final sampling rate. Default 16000.
        filter_order:
            Chebyshev Type-I filter order.
        filter_ripple:
            Chebyshev Type-I passband ripple.

    Returns:
        degraded waveforms:
            [B, T_max], sampled at 16 kHz, but information-limited by
            the intermediate downsample_sr.
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

        filtered = sosfiltfilt(sos, wav)
        down = resample_poly(filtered, downsample_sr, source_sr)
        up = resample_poly(down, source_sr, downsample_sr)

        if up.shape[0] > length:
            up = up[:length]
        elif up.shape[0] < length:
            up = np.pad(up, (0, length - up.shape[0]))

        out[i, :length] = up.astype(np.float32)

    return torch.from_numpy(out).to(device=device, dtype=dtype)


def make_train_valid_indices(
    dataset_size: int,
    valid_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be between 0 and 1, got {valid_ratio}")

    indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(indices)

    valid_size = max(1, int(round(dataset_size * valid_ratio)))
    valid_indices = sorted(indices[:valid_size])
    train_indices = sorted(indices[valid_size:])

    if len(train_indices) == 0:
        raise RuntimeError("Empty training split. Decrease valid_ratio.")

    return train_indices, valid_indices


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_loss_curve(
    train_losses: list[float],
    valid_losses: list[float],
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, valid_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Band-Limited Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_accuracy_curve(
    train_accs: list[float],
    valid_accs: list[float],
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(train_accs) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, marker="o", label="Train Accuracy")
    plt.plot(epochs, valid_accs, marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Band-Limited Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def compute_frame_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = PAD_ID,
) -> tuple[int, int]:
    pred = logits.argmax(dim=-1)
    mask = labels != ignore_index

    correct = ((pred == labels) & mask).sum().item()
    total = mask.sum().item()

    return int(correct), int(total)


def run_epoch(
    model: HubertFlowHighPhonemeClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    downsample_sr: int,
    rms_normalize: bool,
    target_rms: float,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_frames = 0

    pbar = tqdm(loader, leave=False)

    for batch in pbar:
        waveforms = batch["waveforms"].to(device)
        waveform_lengths = batch["waveform_lengths"].to(device)
        labels = batch["labels"].to(device)

        # Create band-limited waveform but keep HuBERT input at 16 kHz.
        waveforms = downsample_then_upsample_keep_16k(
            waveforms=waveforms,
            waveform_lengths=waveform_lengths,
            downsample_sr=downsample_sr,
            source_sr=SOURCE_SR,
            filter_order=FILTER_ORDER,
            filter_ripple=FILTER_RIPPLE,
        )

        # Apply RMS normalization after degradation.
        # This is recommended because low-pass/downsample/upsample changes energy.
        if rms_normalize:
            waveforms = rms_normalize_batch(
                waveforms=waveforms,
                waveform_lengths=waveform_lengths,
                target_rms=target_rms,
            )

        target_num_frames = labels.shape[1]

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(
                waveforms=waveforms,
                waveform_lengths=waveform_lengths,
                target_num_frames=target_num_frames,
            )

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=PAD_ID,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        else:
            with torch.no_grad():
                logits, _ = model(
                    waveforms=waveforms,
                    waveform_lengths=waveform_lengths,
                    target_num_frames=target_num_frames,
                )

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=PAD_ID,
                )

        batch_frames = (labels != PAD_ID).sum().item()
        correct, total = compute_frame_accuracy(logits, labels, ignore_index=PAD_ID)

        total_loss += loss.item() * batch_frames
        total_correct += correct
        total_frames += total

        avg_loss = total_loss / max(total_frames, 1)
        avg_acc = total_correct / max(total_frames, 1)

        pbar.set_description(
            f"{'train_bl' if train else 'valid_bl'} "
            f"loss={avg_loss:.4f} acc={avg_acc:.4f}"
        )

    avg_loss = total_loss / max(total_frames, 1)
    avg_acc = total_correct / max(total_frames, 1)

    return avg_loss, avg_acc


def save_checkpoint(
    path: str | Path,
    model: HubertFlowHighPhonemeClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    history: dict[str, list[float]],
    best_valid_acc: float,
    best_valid_loss: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "history": history,
        "best_valid_acc": best_valid_acc,
        "best_valid_loss": best_valid_loss,
        "phones": PHONES_39,
        "num_phonemes": len(PHONES_39),
        "training_input_type": "bandlimited_downsample_then_upsample_keep_16k",
        "source_sample_rate": SOURCE_SR,
        "downsample_sr": args.downsample_sr,
        "lowpass_cutoff_hz": args.downsample_sr / 2.0,
        "filter_order": FILTER_ORDER,
        "filter_ripple": FILTER_RIPPLE,
        "peak_normalize_before_degradation": False,
        "rms_normalize": args.rms_normalize,
        "target_rms": args.target_rms,
    }

    torch.save(ckpt, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train HuBERT frame-level phoneme classifier aligned to FlowHigh mel frames "
            "using band-limited TIMIT waveforms."
        )
    )

    parser.add_argument(
        "--timit_root",
        type=str,
        required=True,
        help="TIMIT root",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints, curves, and split files.",
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument(
        "--downsample_sr",
        type=int,
        default=1200,
        help=(
            "Intermediate low sampling rate used for band-limited training. "
            "Pipeline: 16 kHz -> low-pass -> downsample_sr -> 16 kHz. "
            "Cutoff is downsample_sr / 2."
        ),
    )

    parser.add_argument(
        "--rms_normalize",
        type=str2bool,
        default=True,
        help="Whether to apply length-aware RMS normalization after band-limiting.",
    )
    parser.add_argument(
        "--target_rms",
        type=float,
        default=0.05,
        help="Target RMS value used when --rms_normalize true.",
    )

    parser.add_argument(
        "--freeze_hubert",
        type=str2bool,
        default=False,
        help="Whether to freeze HuBERT. Use false for fine-tuning.",
    )

    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.10,
        help="Validation ratio from official TIMIT TRAIN split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fixed TRAIN/validation split.",
    )

    parser.add_argument(
        "--drop_sa",
        type=str2bool,
        default=False,
        help="Whether to drop SA1/SA2. Default false because user wants to keep them.",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional debug limit for train subset.",
    )
    parser.add_argument(
        "--max_valid_samples",
        type=int,
        default=None,
        help="Optional debug limit for validation subset.",
    )

    parser.add_argument(
        "--hop_length",
        type=int,
        default=256,
        help="FlowHigh mel hop length.",
    )
    parser.add_argument(
        "--win_length",
        type=int,
        default=1024,
        help="FlowHigh mel window length.",
    )
    parser.add_argument(
        "--center",
        type=str2bool,
        default=True,
        help="Whether FlowHigh mel STFT uses center=True.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume from.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_train = TIMITFramePhonemeDataset(
        timit_root=args.timit_root,
        split="TRAIN",
        max_samples=None,
        hop_length=args.hop_length,
        win_length=args.win_length,
        center=args.center,
        drop_sa=args.drop_sa,
    )

    train_indices, valid_indices = make_train_valid_indices(
        dataset_size=len(full_train),
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    if args.max_train_samples is not None:
        train_indices = train_indices[: args.max_train_samples]

    if args.max_valid_samples is not None:
        valid_indices = valid_indices[: args.max_valid_samples]

    split_info = {
        "timit_root": str(args.timit_root),
        "source_split": "TRAIN",
        "test_used_during_training": False,
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "drop_sa": args.drop_sa,
        "num_total_train_split_utterances": len(full_train),
        "num_train_utterances": len(train_indices),
        "num_valid_utterances": len(valid_indices),
        "train_indices": train_indices,
        "valid_indices": valid_indices,
        "training_input_type": "bandlimited_downsample_then_upsample_keep_16k",
        "source_sample_rate": SOURCE_SR,
        "downsample_sr": args.downsample_sr,
        "lowpass_cutoff_hz": args.downsample_sr / 2.0,
        "filter_order": FILTER_ORDER,
        "filter_ripple": FILTER_RIPPLE,
        "peak_normalize_before_degradation": False,
        "rms_normalize": args.rms_normalize,
        "target_rms": args.target_rms,
    }
    save_json(split_info, save_dir / "train_valid_split.json")

    print(
        f"Using official TIMIT TRAIN only: "
        f"{len(train_indices)} train / {len(valid_indices)} validation utterances. "
        f"TEST is not used during training."
    )
    print(f"SA1/SA2 excluded: {args.drop_sa}")
    print(f"Number of phoneme labels: {len(PHONES_39)}")
    print(f"Device: {device}")
    print("Training input type: bandlimited_downsample_then_upsample_keep_16k")
    print(f"Source/model sample rate: {SOURCE_SR}")
    print(f"Intermediate downsample_sr: {args.downsample_sr}")
    print(f"Low-pass cutoff: {args.downsample_sr / 2.0} Hz")
    print(f"RMS normalization after degradation: {args.rms_normalize}")
    print(f"Target RMS: {args.target_rms}")

    train_set = Subset(full_train, train_indices)
    valid_set = Subset(full_train, valid_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_timit_batch,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_timit_batch,
    )

    model = HubertFlowHighPhonemeClassifier(
        num_phonemes=len(PHONES_39),
        freeze_hubert=args.freeze_hubert,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
    }

    start_epoch = 1
    best_valid_acc = -1.0
    best_valid_loss = float("inf")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        history = ckpt.get("history", history)
        best_valid_acc = float(ckpt.get("best_valid_acc", best_valid_acc))
        best_valid_loss = float(ckpt.get("best_valid_loss", best_valid_loss))

        print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            downsample_sr=args.downsample_sr,
            rms_normalize=args.rms_normalize,
            target_rms=args.target_rms,
        )

        valid_loss, valid_acc = run_epoch(
            model=model,
            loader=valid_loader,
            optimizer=None,
            device=device,
            train=False,
            downsample_sr=args.downsample_sr,
            rms_normalize=args.rms_normalize,
            target_rms=args.target_rms,
        )

        history["train_loss"].append(float(train_loss))
        history["valid_loss"].append(float(valid_loss))
        history["train_acc"].append(float(train_acc))
        history["valid_acc"].append(float(valid_acc))

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}"
        )

        save_json(history, save_dir / "history.json")

        save_loss_curve(
            train_losses=history["train_loss"],
            valid_losses=history["valid_loss"],
            save_path=save_dir / "loss_curve.png",
        )

        save_accuracy_curve(
            train_accs=history["train_acc"],
            valid_accs=history["valid_acc"],
            save_path=save_dir / "accuracy_curve.png",
        )

        save_checkpoint(
            path=save_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            history=history,
            best_valid_acc=best_valid_acc,
            best_valid_loss=best_valid_loss,
        )

        improved = False

        if valid_acc > best_valid_acc:
            improved = True
        elif valid_acc == best_valid_acc and valid_loss < best_valid_loss:
            improved = True

        if improved:
            best_valid_acc = float(valid_acc)
            best_valid_loss = float(valid_loss)

            save_checkpoint(
                path=save_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                history=history,
                best_valid_acc=best_valid_acc,
                best_valid_loss=best_valid_loss,
            )

            print(
                f"Saved best checkpoint: valid_acc={best_valid_acc:.4f}, "
                f"valid_loss={best_valid_loss:.4f}"
            )

    print("\nTraining finished.")
    print(f"Best valid acc: {best_valid_acc:.4f}")
    print(f"Best valid loss: {best_valid_loss:.4f}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()