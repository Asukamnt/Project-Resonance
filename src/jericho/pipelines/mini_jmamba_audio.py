"""Mini-JMamba audio training pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from jericho.data import ManifestEntry, synthesise_entry_wave
from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.scorer import decode_wave_to_symbols, exact_match


@dataclass
class SymbolVocab:
    symbol_to_id: dict[str, int]
    id_to_symbol: list[str]
    blank_id: int = 0

    @property
    def size(self) -> int:
        return len(self.id_to_symbol)


@dataclass
class MiniJMambaTrainingConfig:
    frame_size: int = 160
    hop_size: int = 160
    d_model: int = 128
    num_ssm_layers: int = 10
    num_attn_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1
    ctc_weight: float = 0.3


@dataclass
class AudioSample:
    entry: ManifestEntry
    frames: torch.Tensor
    frame_count: int
    wave: torch.Tensor
    target_ids: List[int]


class AudioFrameDataset(Dataset):
    """Dataset of framed audio with symbol targets."""

    def __init__(self, samples: Sequence[AudioSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> AudioSample:
        return self.samples[idx]

    @property
    def frame_counts(self) -> List[int]:
        return [sample.frame_count for sample in self.samples]

    @property
    def entries(self) -> List[ManifestEntry]:
        return [sample.entry for sample in self.samples]


def build_symbol_vocab(entries: Sequence[ManifestEntry]) -> SymbolVocab:
    symbols = set()
    for entry in entries:
        symbols.update(entry.symbols)
    ordered = sorted(symbols)
    symbol_to_id = {symbol: idx + 1 for idx, symbol in enumerate(ordered)}
    id_to_symbol = ["<blank>"] + ordered
    return SymbolVocab(symbol_to_id=symbol_to_id, id_to_symbol=id_to_symbol, blank_id=0)


def frame_wave(wave: torch.Tensor, frame_size: int, hop_size: int) -> torch.Tensor:
    length = wave.size(0)
    num_frames = int(np.ceil(length / hop_size))
    frames = torch.zeros(num_frames, frame_size, dtype=wave.dtype)
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        segment = wave[start:end]
        frames[i, : segment.size(0)] = segment
    return frames


def prepare_audio_samples(
    entries: Sequence[ManifestEntry],
    vocab: SymbolVocab,
    frame_size: int,
    hop_size: int,
) -> List[AudioSample]:
    samples: list[AudioSample] = []
    for entry in entries:
        wave_np = synthesise_entry_wave(entry)
        wave = torch.from_numpy(np.asarray(wave_np, dtype=np.float32))
        frames = frame_wave(wave, frame_size, hop_size)
        target_ids = [vocab.symbol_to_id[s] for s in entry.symbols if s in vocab.symbol_to_id]
        samples.append(
            AudioSample(
                entry=entry,
                frames=frames,
                frame_count=frames.size(0),
                wave=wave,
                target_ids=target_ids,
            )
        )
    return samples


def collate_train(
    batch: Sequence[AudioSample],
    frame_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_frames = max(sample.frame_count for sample in batch)
    features = torch.zeros(len(batch), max_frames, frame_size, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_frames, dtype=torch.bool)
    target_lengths = []
    all_targets: list[torch.Tensor] = []
    wave_lengths = []
    for idx, sample in enumerate(batch):
        features[idx, : sample.frame_count] = sample.frames
        mask[idx, : sample.frame_count] = True
        target_lengths.append(len(sample.target_ids))
        if sample.target_ids:
            all_targets.append(torch.tensor(sample.target_ids, dtype=torch.long))
        wave_lengths.append(sample.wave.size(0))
    if all_targets:
        targets = torch.cat(all_targets)
    else:
        targets = torch.zeros(0, dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)
    wave_lengths_tensor = torch.tensor(wave_lengths, dtype=torch.long)
    return features, mask, targets, target_lengths_tensor, wave_lengths_tensor


def collate_eval(
    batch: Sequence[AudioSample],
    frame_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_frames = max(sample.frame_count for sample in batch)
    features = torch.zeros(len(batch), max_frames, frame_size, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_frames, dtype=torch.bool)
    wave_lengths = []
    for idx, sample in enumerate(batch):
        features[idx, : sample.frame_count] = sample.frames
        mask[idx, : sample.frame_count] = True
        wave_lengths.append(sample.wave.size(0))
    wave_lengths_tensor = torch.tensor(wave_lengths, dtype=torch.long)
    return features, mask, wave_lengths_tensor


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    diff = diff * mask.unsqueeze(-1)
    denom = mask.sum() * target.size(-1)
    denom = denom.clamp_min(1)
    return diff.sum() / denom


def evaluate_model(
    model: MiniJMamba,
    dataloader: DataLoader,
    dataset: AudioFrameDataset,
    device: torch.device,
    *,
    collect_predictions: bool = False,
) -> Tuple[float, float, List[dict]]:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    matches = 0
    predictions: list[dict] = []
    entry_idx = 0
    with torch.no_grad():
        for features, mask, wave_lengths in dataloader:
            features = features.to(device)
            mask = mask.to(device)
            frame_outputs, _ = model(features, mask)
            loss = masked_mse(frame_outputs, features, mask)
            total_loss += loss.item() * mask.sum().item() * features.size(-1)
            total_weight += mask.sum().item() * features.size(-1)

            outputs_np = frame_outputs.cpu().numpy()
            wave_lengths_np = wave_lengths.numpy()
            for b in range(features.size(0)):
                entry = dataset.entries[entry_idx]
                frames_count = dataset.samples[entry_idx].frame_count
                wave_len = wave_lengths_np[b]
                frames_flat = outputs_np[b, :frames_count].reshape(-1)
                pred_wave = frames_flat[:wave_len]
                pred_wave = np.clip(pred_wave, -1.0, 1.0).astype(np.float32)
                pred_symbols = decode_wave_to_symbols(pred_wave)
                em = exact_match(pred_symbols, entry.symbols)
                if em == 1.0:
                    matches += 1
                if collect_predictions:
                    predictions.append(
                        {
                            "example_id": entry.example_id,
                            "split": entry.split,
                            "gold_symbols": entry.symbols,
                            "pred_symbols": pred_symbols,
                            "exact_match": em,
                        }
                    )
                entry_idx += 1

    avg_loss = total_loss / max(1.0, total_weight)
    em = matches / len(dataset) if len(dataset) else 0.0
    return avg_loss, em, predictions


def mini_jmamba_pipeline(
    train_entries: Sequence[ManifestEntry],
    eval_entries: Sequence[ManifestEntry],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    config: MiniJMambaTrainingConfig | None = None,
) -> Tuple[List[dict], dict]:
    """Train and evaluate Mini-JMamba on framed audio data."""

    if config is None:
        config = MiniJMambaTrainingConfig()

    torch.manual_seed(seed)
    np.random.seed(seed)

    vocab = build_symbol_vocab(train_entries)
    train_samples = prepare_audio_samples(
        train_entries, vocab, config.frame_size, config.hop_size
    )
    eval_samples = prepare_audio_samples(
        eval_entries, vocab, config.frame_size, config.hop_size
    )
    train_dataset = AudioFrameDataset(train_samples)
    eval_dataset = AudioFrameDataset(eval_samples)

    if len(train_dataset) == 0:
        raise SystemExit("Train split is empty; cannot train mini_jmamba.")
    if len(eval_dataset) == 0:
        raise SystemExit("Evaluation split is empty.")

    max_frames = max(max(train_dataset.frame_counts), max(eval_dataset.frame_counts))
    model_config = MiniJMambaConfig(
        frame_size=config.frame_size,
        hop_size=config.hop_size,
        symbol_vocab_size=vocab.size,
        d_model=config.d_model,
        num_ssm_layers=config.num_ssm_layers,
        num_attn_layers=config.num_attn_layers,
        num_heads=config.num_heads,
        max_frames=max_frames,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
    )
    model = MiniJMamba(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss_fn = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_train(batch, config.frame_size),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_eval(batch, config.frame_size),
    )

    loss_pre, em_pre, _ = evaluate_model(
        model,
        eval_loader,
        eval_dataset,
        device=device,
        collect_predictions=False,
    )
    print(f"[mini_jmamba] pre-training loss={loss_pre:.6f} em={em_pre:.4f}")

    for epoch in range(epochs):
        model.train()
        for features, mask, targets, target_lengths, _ in train_loader:
            features = features.to(device)
            mask = mask.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            frame_outputs, symbol_logits = model(features, mask)
            audio_loss = masked_mse(frame_outputs, features, mask)

            total_loss = audio_loss
            if config.ctc_weight > 0 and targets.numel() > 0:
                symbol_logits = symbol_logits.masked_fill(~mask.unsqueeze(-1), 0.0)
                log_probs = symbol_logits.log_softmax(dim=-1).permute(1, 0, 2)
                input_lengths = mask.sum(dim=1).to(torch.long)
                ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                total_loss = total_loss + config.ctc_weight * ctc_loss
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  epoch {epoch + 1}/{epochs} - train_loss={total_loss.item():.6f}")

    loss_post, em_post, predictions = evaluate_model(
        model,
        eval_loader,
        eval_dataset,
        device=device,
        collect_predictions=True,
    )
    print(f"[mini_jmamba] post-training loss={loss_post:.6f} em={em_post:.4f}")
    for sample in predictions[:3]:
        gold = " ".join(sample["gold_symbols"])
        pred = " ".join(sample["pred_symbols"])
        print(
            f"example {sample['example_id']}: gold=[{gold}] pred=[{pred}] em={sample['exact_match']:.1f}"
        )

    metrics = {
        "loss_pre": loss_pre,
        "loss_post": loss_post,
        "em_pre": em_pre,
        "em_post": em_post,
        "model_config": asdict(model_config),
        "vocab": {"symbol_to_id": vocab.symbol_to_id, "blank_id": vocab.blank_id},
        "ctc_weight": config.ctc_weight,
    }
    return predictions, metrics


__all__ = [
    "MiniJMambaTrainingConfig",
    "mini_jmamba_pipeline",
    "SymbolVocab",
    "frame_wave",
    "masked_mse",
]

