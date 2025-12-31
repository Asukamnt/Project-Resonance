"""Pipeline entrypoint for Stage A/B baselines and Mini-JMamba."""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from jericho.baselines import predict_wave_identity, predict_wave_oracle_mod
from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.pipelines import (
    MiniJMambaTrainingConfig,
    Task2TrainingConfig,
    Task3TrainingConfig,
    mini_jmamba_pipeline,
    mini_jmamba_task2_pipeline,
    mini_jmamba_task3_pipeline,
)
from jericho.scorer import decode_wave_to_symbols, exact_match
from jericho.task2 import target_symbol_for_task2
from jericho.task3 import target_symbols_for_task3

SplitName = str


def _parse_yaml_scalar(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    # Numbers
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        pass
    # Quoted strings
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    return text


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load a minimal YAML subset: top-level key/value scalars.

    Supported:
    - `key: value` pairs (no nesting)
    - comments starting with `#`
    - scalars: int/float/bool/null/str
    """

    data: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"Invalid YAML at {path}:{line_no}: expected 'key: value'")
            key_raw, value_raw = line.split(":", 1)
            key = key_raw.strip()
            if not key:
                raise ValueError(f"Invalid YAML at {path}:{line_no}: empty key")
            # Strip inline comments (simple heuristic: only when value is unquoted)
            value_part = value_raw.strip()
            if value_part and not (value_part.startswith('"') or value_part.startswith("'")):
                value_part = value_part.split("#", 1)[0].strip()
            data[key] = _parse_yaml_scalar(value_part)
    return data


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("JSON config must be an object at top-level")
        return payload
    if suffix in {".yaml", ".yml"}:
        return _load_simple_yaml(path)
    raise ValueError(f"Unsupported config format: {path.suffix} (expected .yaml/.yml/.json)")


def _apply_config_defaults(parser: argparse.ArgumentParser, config: dict[str, Any]) -> None:
    # Normalise keys (allow CLI-style kebab-case in config too).
    normalised: dict[str, Any] = {str(k).replace("-", "_"): v for k, v in config.items()}

    valid_dests = {action.dest for action in parser._actions if action.dest != "help"}  # noqa: SLF001
    unknown = sorted([k for k in normalised.keys() if k not in valid_dests])
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    defaults: dict[str, Any] = {}
    for action in parser._actions:  # noqa: SLF001
        dest = action.dest
        if dest == "help" or dest not in normalised:
            continue
        value = normalised[dest]

        # Apply basic type conversions for defaults (argparse does not type-cast defaults).
        if value is None:
            converted = None
        elif isinstance(action, argparse.BooleanOptionalAction):
            if isinstance(value, bool):
                converted = value
            elif isinstance(value, str):
                converted = _parse_yaml_scalar(value)
                if not isinstance(converted, bool):
                    raise ValueError(f"Config key '{dest}' must be boolean")
            else:
                raise ValueError(f"Config key '{dest}' must be boolean")
        elif action.type is Path:
            converted = Path(str(value))
        elif action.type is int:
            converted = int(value)
        elif action.type is float:
            converted = float(value)
        elif action.type is str:
            converted = str(value)
        else:
            converted = value

        if getattr(action, "choices", None) is not None and converted is not None:
            if converted not in action.choices:
                raise ValueError(
                    f"Invalid value for config key '{dest}': {converted!r} (choices={list(action.choices)!r})"
                )

        defaults[dest] = converted

    parser.set_defaults(**defaults)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage A/B training pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON config file. CLI args override values from this file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/task1.jsonl"),
        help="Manifest path.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for run artifacts (defaults to runs/<model>_<timestamp>).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="identity",
        choices=["identity", "mini_jmamba", "oracle_mod", "transformer", "lstm"],
        help="Model/baseline to execute. transformer/lstm only supported for task=mod.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mirror",
        choices=["mirror", "bracket", "mod"],
        help="Task to run (mirror=Task1 copy, bracket=Task2 括号匹配, mod=Task3 modulo)。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "iid_test", "ood_length", "ood_symbol", "ood_digits", "ood_noise", "ood_compose"],
        help="Evaluation split (训练始终使用 train split)。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit applied to both training and evaluation splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for reproducibility.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for models.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    # Model architecture parameters (for scaling experiments)
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model hidden dimension (d_model). Larger = more capacity.",
    )
    parser.add_argument(
        "--num-ssm-layers",
        type=int,
        default=10,
        help="Number of SSM layers in Mini-JMamba.",
    )
    parser.add_argument(
        "--num-attn-layers",
        type=int,
        default=2,
        help="Number of attention layers in Mini-JMamba.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run training/evaluation on.",
    )
    parser.add_argument(
        "--pretrain-mirror-epochs",
        type=int,
        default=Task3TrainingConfig().pretrain_mirror_epochs,
        help="Number of mirror pretrain epochs before mod training (Task3).",
    )
    parser.add_argument(
        "--pretrain-mirror-ctc-weight",
        type=float,
        default=Task3TrainingConfig().pretrain_mirror_ctc_weight,
        help="CTC weight during mirror pretraining (Task3).",
    )
    parser.add_argument(
        "--pretrain-mirror-frame-ce-weight",
        type=float,
        default=Task3TrainingConfig().pretrain_mirror_frame_ce_weight,
        help="Frame CE weight during mirror pretraining (Task3).",
    )
    parser.add_argument(
        "--pretrain-mirror-blank-penalty-weight",
        type=float,
        default=Task3TrainingConfig().pretrain_mirror_blank_penalty_weight,
        help="Blank penalty weight during mirror pretraining (Task3).",
    )
    parser.add_argument(
        "--pretrain-mirror-audio-weight",
        type=float,
        default=Task3TrainingConfig().pretrain_mirror_audio_weight,
        help="Audio loss weight during mirror pretraining (Task3).",
    )
    parser.add_argument(
        "--pretrain-mirror-answer-window-only",
        action=argparse.BooleanOptionalAction,
        default=Task3TrainingConfig().pretrain_mirror_answer_window_only,
        help="If set, restrict mirror pretraining losses to expression window only.",
    )
    parser.add_argument(
        "--mod-expr-ctc-weight-start",
        type=float,
        default=1.0,
        help="Starting weight for expression CTC during mod training (anti-forgetting).",
    )
    parser.add_argument(
        "--mod-expr-ctc-weight-end",
        type=float,
        default=0.2,
        help="Ending weight for expression CTC during mod training (anti-forgetting).",
    )
    parser.add_argument(
        "--mod-lr-factor",
        type=float,
        default=0.1,
        help="Learning rate multiplier applied when entering mod stage (anti-forgetting).",
    )
    parser.add_argument(
        "--thinking-gap-s",
        type=float,
        default=0.5,
        help="Thinking gap duration in seconds before answer window (Task3).",
    )
    parser.add_argument(
        "--thinking-gap-align",
        type=int,
        default=160,
        help="Alignment (in samples) for thinking gap and answer window (Task3).",
    )
    parser.add_argument(
        "--remainder-guidance-weight",
        type=float,
        default=Task3TrainingConfig().remainder_guidance_weight,
        help="Weight for remainder guidance mixing in answer window.",
    )
    parser.add_argument(
        "--remainder-guidance-blank-floor",
        type=float,
        default=Task3TrainingConfig().remainder_guidance_blank_floor,
        help="Floor mass for blank/non-digit tokens in answer guidance mix.",
    )
    parser.add_argument(
        "--answer-digit-mass-floor",
        type=float,
        default=Task3TrainingConfig().answer_digit_mass_floor,
        help="Minimum digit mass enforced when rendering answer window tone bank.",
    )
    parser.add_argument(
        "--answer-blank-margin",
        type=float,
        default=Task3TrainingConfig().answer_blank_margin,
        help="Margin between digit logits and blank logits on answer window tones.",
    )
    parser.add_argument(
        "--answer-blank-margin-weight",
        type=float,
        default=Task3TrainingConfig().answer_blank_margin_weight,
        help="Weight for answer-window blank suppression margin loss.",
    )
    parser.add_argument(
        "--remainder-head",
        type=str,
        default=Task3TrainingConfig().remainder_head,
        choices=["attn_hidden", "gru_token", "gru_frame", "pooled"],
        help=(
            "Architecture for remainder prediction head "
            "(attn_hidden=attention pooling over backbone hidden states, "
            "gru_token=token-level GRU, gru_frame=frame-level GRU, pooled=masked mean pooling)."
        ),
    )
    parser.add_argument(
        "--remainder-gru-hidden",
        type=int,
        default=Task3TrainingConfig().remainder_gru_hidden,
        help="Hidden size for GRU remainder head (when remainder-head=gru).",
    )
    parser.add_argument(
        "--pretrain-remainder-epochs",
        type=int,
        default=Task3TrainingConfig().pretrain_remainder_epochs,
        help="Remainder-only pretrain epochs after mirror pretrain and before mod training.",
    )
    parser.add_argument(
        "--pretrain-remainder-lr",
        type=float,
        default=Task3TrainingConfig().pretrain_remainder_lr,
        help="Learning rate for remainder-only pretraining.",
    )
    parser.add_argument(
        "--pretrain-remainder-freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=Task3TrainingConfig().pretrain_remainder_freeze_backbone,
        help="Freeze backbone during remainder-only pretraining.",
    )
    parser.add_argument(
        "--blank-penalty-weight",
        type=float,
        default=Task3TrainingConfig().blank_penalty_weight,
        help="Penalty weight for blank predictions in answer window.",
    )
    parser.add_argument(
        "--symbol-warmup-epochs",
        type=int,
        default=Task3TrainingConfig().symbol_warmup_epochs,
        help="Number of epochs to delay/anneal audio losses.",
    )
    # Task2 specific arguments
    parser.add_argument(
        "--task2-binary-ce-weight",
        type=float,
        default=Task2TrainingConfig().binary_ce_weight,
        help="Weight for binary classification CE loss (Task2 bracket).",
    )
    parser.add_argument(
        "--task2-symbol-guidance-weight",
        type=float,
        default=Task2TrainingConfig().symbol_guidance_weight,
        help="Weight for symbol guidance in audio generation (Task2 bracket).",
    )
    parser.add_argument(
        "--task2-thinking-gap-s",
        type=float,
        default=Task2TrainingConfig().thinking_gap_s,
        help="Thinking gap duration in seconds for Task2 bracket.",
    )
    parser.add_argument(
        "--task2-answer-window-only",
        action=argparse.BooleanOptionalAction,
        default=Task2TrainingConfig().answer_window_only,
        help="If set, compute Task2 audio reconstruction losses only on the answer window.",
    )
    parser.add_argument(
        "--task2-noise-snr",
        type=float,
        default=None,
        help="If set, add Gaussian noise at this SNR (dB) to eval inputs. Use for OOD noise testing.",
    )
    parser.add_argument(
        "--task2-audio-ramp-epochs",
        type=int,
        default=Task2TrainingConfig().audio_ramp_epochs,
        help="Number of epochs to ramp Task2 audio loss after warmup.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # Pre-parse to locate config file (so we can apply defaults before full parsing).
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    known, _ = config_parser.parse_known_args(argv)

    parser = _build_parser()
    if known.config is not None:
        try:
            cfg = _load_config_file(known.config)
            _apply_config_defaults(parser, cfg)
        except Exception as exc:
            raise SystemExit(f"Failed to load config '{known.config}': {exc}") from exc

    return parser.parse_args(argv)


def resolve_outdir(base: Path | None, model: str, task: str) -> Path:
    if base is not None:
        return base
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{model}_{task}_{timestamp}"


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def summarise_predictions(predictions: Iterable[dict]) -> Dict[SplitName, dict]:
    summary: Dict[SplitName, dict] = {}
    for pred in predictions:
        split = pred["split"]
        stats = summary.setdefault(split, {"matches": 0.0, "count": 0})
        stats["matches"] += pred["exact_match"]
        stats["count"] += 1
    return summary


def run_identity_baseline(
    entries: Sequence[ManifestEntry],
    *,
    seed: int,
) -> List[dict]:
    results: list[dict] = []
    for entry in entries:
        input_wave = synthesise_entry_wave(entry)
        pred_wave = predict_wave_identity(input_wave, rng_seed=seed)
        pred_symbols = decode_wave_to_symbols(pred_wave)
        em = exact_match(pred_symbols, entry.symbols)
        results.append(
            {
                "example_id": entry.example_id,
                "split": entry.split,
                "gold_symbols": entry.symbols,
                "pred_symbols": pred_symbols,
                "exact_match": em,
            }
        )
    return results


def identity_pipeline(
    entries: Sequence[ManifestEntry],
    *,
    seed: int,
) -> Tuple[List[dict], dict]:
    predictions = run_identity_baseline(entries, seed=seed)
    summary = summarise_predictions(predictions)
    overall_matches = sum(stats["matches"] for stats in summary.values())
    overall_count = sum(stats["count"] for stats in summary.values())
    overall_em = overall_matches / overall_count if overall_count else 0.0

    for record in predictions[: min(3, len(predictions))]:
        gold = " ".join(record["gold_symbols"])
        pred = " ".join(record["pred_symbols"])
        print(f"example {record['example_id']}: gold=[{gold}] pred=[{pred}] em={record['exact_match']:.1f}")

    for split, stats in summary.items():
        em = stats["matches"] / stats["count"] if stats["count"] else 0.0
        print(f"[{split}] Exact Match: {em:.4f} ({stats['count']} samples)")
    print(f"[overall] Exact Match: {overall_em:.4f} ({overall_count} samples)")

    metrics = {"count": overall_count, "exact_match": overall_em}
    return predictions, metrics


def oracle_mod_pipeline(entries: Sequence[ManifestEntry]) -> Tuple[List[dict], dict]:
    predictions: list[dict] = []
    for entry in entries:
        input_wave = synthesise_entry_wave(entry)
        pred_wave = predict_wave_oracle_mod(input_wave)
        pred_symbols = decode_wave_to_symbols(pred_wave)
        gold_symbols = target_symbols_for_task3(entry.symbols)
        em = exact_match(pred_symbols, gold_symbols)
        predictions.append(
            {
                "example_id": entry.example_id,
                "split": entry.split,
                "gold_symbols": gold_symbols,
                "pred_symbols": pred_symbols,
                "exact_match": em,
            }
        )
    for record in predictions[: min(3, len(predictions))]:
        gold = " ".join(record["gold_symbols"])
        pred = " ".join(record["pred_symbols"])
        print(f"example {record['example_id']}: gold=[{gold}] pred=[{pred}] em={record['exact_match']:.1f}")
    exact = sum(p["exact_match"] for p in predictions) / max(1, len(predictions))
    metrics = {"count": len(predictions), "exact_match": exact}
    return predictions, metrics


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    outdir = resolve_outdir(args.outdir, args.model, args.task)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.task == "mirror":
        if args.model not in {"identity", "mini_jmamba"}:
            raise SystemExit(f"Model '{args.model}' unsupported for task mirror.")
        eval_entries = read_manifest(args.manifest, split=args.split)
        if not eval_entries:
            raise SystemExit(f"No entries found for split='{args.split}' in {args.manifest}")
        if args.limit is not None:
            eval_entries = eval_entries[: args.limit]

        if args.model == "identity":
            predictions, model_metrics = identity_pipeline(eval_entries, seed=args.seed)
        else:
            train_entries = read_manifest(args.manifest, split="train")
            if not train_entries:
                raise SystemExit("Train split is empty; cannot train mini_jmamba.")
            if args.limit is not None:
                train_entries = train_entries[: args.limit]
            predictions, model_metrics = mini_jmamba_pipeline(
                train_entries,
                eval_entries,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=torch.device(args.device),
                config=MiniJMambaTrainingConfig(),
            )
    elif args.task == "bracket":  # Task2 bracket validity
        if args.model != "mini_jmamba":
            raise SystemExit(f"Model '{args.model}' unsupported for task bracket (only mini_jmamba).")
        eval_entries = read_manifest(args.manifest, split=args.split)
        if not eval_entries:
            raise SystemExit(f"No entries found for split='{args.split}' in {args.manifest}")
        if args.limit is not None:
            eval_entries = eval_entries[: args.limit]

        train_entries = read_manifest(args.manifest, split="train")
        if not train_entries:
            raise SystemExit("Train split is empty; cannot train mini_jmamba on bracket.")
        if args.limit is not None:
            train_entries = train_entries[: args.limit]
        # Auto-detect noise SNR for ood_noise split
        eval_noise_snr = args.task2_noise_snr
        if eval_noise_snr is None and args.split == "ood_noise":
            eval_noise_snr = 10.0  # Default 10 dB SNR for OOD noise

        predictions, model_metrics = mini_jmamba_task2_pipeline(
            train_entries,
            eval_entries,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=torch.device(args.device),
            config=Task2TrainingConfig(
                binary_ce_weight=args.task2_binary_ce_weight,
                symbol_guidance_weight=args.task2_symbol_guidance_weight,
                thinking_gap_s=args.task2_thinking_gap_s,
                symbol_warmup_epochs=args.symbol_warmup_epochs,
                audio_ramp_epochs=args.task2_audio_ramp_epochs,
                answer_window_only=args.task2_answer_window_only,
            ),
            eval_noise_snr_db=eval_noise_snr,
        )
    else:  # Task3 mod
        if args.model not in {"oracle_mod", "mini_jmamba", "transformer", "lstm"}:
            raise SystemExit(f"Model '{args.model}' unsupported for task mod.")
        eval_entries = read_manifest(args.manifest, split=args.split)
        if not eval_entries:
            raise SystemExit(f"No entries found for split='{args.split}' in {args.manifest}")
        if args.limit is not None:
            eval_entries = eval_entries[: args.limit]

        if args.model == "oracle_mod":
            predictions, model_metrics = oracle_mod_pipeline(eval_entries)
        else:
            # Determine backbone: mini_jmamba, transformer, or lstm
            backbone_name = args.model if args.model in {"transformer", "lstm"} else "mini_jmamba"
            
            train_entries = read_manifest(args.manifest, split="train")
            if not train_entries:
                raise SystemExit(f"Train split is empty; cannot train {backbone_name} on mod.")
            if args.limit is not None:
                train_entries = train_entries[: args.limit]
            predictions, model_metrics, model_info = mini_jmamba_task3_pipeline(
                train_entries,
                eval_entries,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=torch.device(args.device),
                config=Task3TrainingConfig(
                    # Model architecture
                    d_model=args.d_model,
                    num_ssm_layers=args.num_ssm_layers,
                    num_attn_layers=args.num_attn_layers,
                    num_heads=args.num_heads,
                    # Training config
                    pretrain_mirror_epochs=args.pretrain_mirror_epochs,
                    pretrain_mirror_ctc_weight=args.pretrain_mirror_ctc_weight,
                    pretrain_mirror_frame_ce_weight=args.pretrain_mirror_frame_ce_weight,
                    pretrain_mirror_blank_penalty_weight=args.pretrain_mirror_blank_penalty_weight,
                    pretrain_mirror_audio_weight=args.pretrain_mirror_audio_weight,
                    pretrain_mirror_answer_window_only=args.pretrain_mirror_answer_window_only,
                    thinking_gap_s=args.thinking_gap_s,
                    thinking_gap_align=args.thinking_gap_align,
                    mod_expr_ctc_weight_start=args.mod_expr_ctc_weight_start,
                    mod_expr_ctc_weight=args.mod_expr_ctc_weight_end,
                    remainder_guidance_weight=args.remainder_guidance_weight,
                    remainder_guidance_blank_floor=args.remainder_guidance_blank_floor,
                    answer_digit_mass_floor=args.answer_digit_mass_floor,
                    answer_blank_margin=args.answer_blank_margin,
                    answer_blank_margin_weight=args.answer_blank_margin_weight,
                    remainder_head=args.remainder_head,
                    remainder_gru_hidden=args.remainder_gru_hidden,
                    pretrain_remainder_epochs=args.pretrain_remainder_epochs,
                    pretrain_remainder_lr=args.pretrain_remainder_lr,
                    pretrain_remainder_freeze_backbone=args.pretrain_remainder_freeze_backbone,
                    blank_penalty_weight=args.blank_penalty_weight,
                    symbol_warmup_epochs=args.symbol_warmup_epochs,
                ),
                mod_lr_factor=args.mod_lr_factor,
                backbone=backbone_name,
            )
            # 保存 checkpoint（包含完整训练参数）
            checkpoint_path = outdir / f"mod_seed{args.seed}_epoch{args.epochs}.pt"
            
            # Build config dict based on model type
            model_cfg = model_info["model_config"]
            if hasattr(model_cfg, "num_ssm_layers"):
                # MiniJMambaConfig
                config_dict = {
                    "frame_size": model_cfg.frame_size,
                    "hop_size": model_cfg.hop_size,
                    "symbol_vocab_size": model_cfg.symbol_vocab_size,
                    "d_model": model_cfg.d_model,
                    "num_ssm_layers": model_cfg.num_ssm_layers,
                    "num_attn_layers": model_cfg.num_attn_layers,
                    "num_heads": model_cfg.num_heads,
                    "max_frames": model_cfg.max_frames,
                    "use_rope": model_cfg.use_rope,
                }
            else:
                # BaselineConfig (Transformer/LSTM)
                config_dict = model_cfg.to_dict()
            
            torch.save({
                "model_state_dict": model_info["model"].state_dict(),
                "backbone": backbone_name,
                "config": config_dict,
                "symbol_to_id": model_info["symbol_to_id"],
                "id_to_symbol": model_info["id_to_symbol"],
                "task": "mod",
                "epochs": args.epochs,
                "seed": args.seed,
                # 重要训练参数
                "training_params": {
                    "manifest": str(args.manifest),
                    "split": args.split,
                    "limit": args.limit,
                    "pretrain_mirror_epochs": args.pretrain_mirror_epochs,
                    "pretrain_remainder_epochs": args.pretrain_remainder_epochs,
                    "blank_penalty_weight": args.blank_penalty_weight,
                    "remainder_guidance_weight": args.remainder_guidance_weight,
                    "symbol_warmup_epochs": args.symbol_warmup_epochs,
                    "mod_lr_factor": args.mod_lr_factor,
                },
                "git_commit": get_git_commit(),
                "cli_argv": sys.argv,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    preds_path = outdir / "preds.jsonl"
    with preds_path.open("w", encoding="utf-8") as f:
        for record in predictions:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    summary = summarise_predictions(predictions)
    overall_matches = sum(stats["matches"] for stats in summary.values())
    overall_count = sum(stats["count"] for stats in summary.values())
    overall_em = overall_matches / overall_count if overall_count else 0.0

    for split, stats in summary.items():
        em = stats["matches"] / stats["count"] if stats["count"] else 0.0
        print(f"[{split}] Exact Match: {em:.4f} ({stats['count']} samples)")
    print(f"[overall] Exact Match: {overall_em:.4f} ({overall_count} samples)")

    metrics: Dict[str, object] = {
        "task": args.task,
        "model": args.model,
        "eval_split": args.split,
        "count": overall_count,
        "exact_match": overall_em,
        "per_split": {
            split: stats["matches"] / stats["count"] if stats["count"] else 0.0
            for split, stats in summary.items()
        },
        "seed": args.seed,
        "manifest": str(args.manifest),
        "limit": args.limit,
        "config_file": str(args.config) if getattr(args, "config", None) else None,
        "cli_argv": sys.argv,
        "git_commit": get_git_commit(),
    }
    metrics.update(model_metrics)

    metrics_path = outdir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions to {preds_path}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()

