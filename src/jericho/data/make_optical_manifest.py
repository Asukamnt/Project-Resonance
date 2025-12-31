"""Manifest generation for Phase2 optical domain tasks.

This module generates manifest files for Task1/Task2/Task3 in the optical
domain, with proper OOD splits and symbol holdout.

OOD-symbol holdout (v0.1):
- Task1: F excluded from iid_train/val, required in ood_symbol
- Task3: 9 excluded from iid_train/val, required in ood_symbol
- Task2: No symbol holdout (uses length/noise/drift OOD axes)

Reference: docs/phase2_light_to_light.md Section 4.3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Literal, Optional, Sequence

from jericho.domains.optical_intensity import (
    OPTICAL_SYMBOL_VOCAB,
    OOD_SYMBOL_HOLDOUT,
)


@dataclass
class OpticalManifestEntry:
    """Manifest entry for optical domain tasks."""
    
    split: str
    symbols: List[str]
    target_symbols: List[str]
    length: int
    task: str
    difficulty_tag: str
    example_id: str
    seed: int
    sequence_seed: int
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# Task1 (Mirror) symbols: A-E for IID, F for OOD
TASK1_IID_SYMBOLS = ["A", "B", "C", "D", "E"]
TASK1_OOD_SYMBOLS = ["F"]
TASK1_ALL_SYMBOLS = TASK1_IID_SYMBOLS + TASK1_OOD_SYMBOLS

# Task2 (Bracket) symbols
TASK2_INPUT_SYMBOLS = ["(", ")"]
TASK2_OUTPUT_SYMBOLS = ["V", "X"]

# Task3 (Mod) symbols: 0-8 for IID, 9 for OOD
TASK3_IID_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
TASK3_OOD_DIGITS = ["9"]
TASK3_ALL_DIGITS = TASK3_IID_DIGITS + TASK3_OOD_DIGITS
TASK3_OPERATOR = "%"


def generate_task1_mirror_entry(
    split: str,
    length: int,
    example_id: str,
    seed: int,
    use_ood_symbols: bool = False,
) -> OpticalManifestEntry:
    """Generate a Task1 (Mirror) manifest entry.
    
    Parameters
    ----------
    split : str
        Split name (iid_train, iid_val, iid_test, ood_length, ood_symbol)
    length : int
        Sequence length
    example_id : str
        Unique example identifier
    seed : int
        Random seed for this example
    use_ood_symbols : bool
        If True, include OOD symbols (F)
    """
    rng = random.Random(seed)
    
    if use_ood_symbols:
        # OOD-symbol: must include at least one F
        pool = TASK1_ALL_SYMBOLS
        symbols = [rng.choice(pool) for _ in range(length - 1)]
        # Insert F at random position
        insert_pos = rng.randint(0, len(symbols))
        symbols.insert(insert_pos, "F")
    else:
        symbols = [rng.choice(TASK1_IID_SYMBOLS) for _ in range(length)]
    
    return OpticalManifestEntry(
        split=split,
        symbols=symbols,
        target_symbols=symbols.copy(),  # Mirror: target = input
        length=length,
        task="mirror",
        difficulty_tag=f"len{length}",
        example_id=example_id,
        seed=seed,
        sequence_seed=seed,
    )


def generate_task2_bracket_entry(
    split: str,
    length: int,
    example_id: str,
    seed: int,
    force_valid: Optional[bool] = None,
) -> OpticalManifestEntry:
    """Generate a Task2 (Bracket) manifest entry.
    
    Parameters
    ----------
    split : str
        Split name
    length : int
        Number of brackets (must be even for valid)
    example_id : str
        Unique example identifier
    seed : int
        Random seed
    force_valid : bool, optional
        If True, generate valid bracket sequence
        If False, generate invalid
        If None, random
    """
    rng = random.Random(seed)
    
    # Determine validity
    if force_valid is None:
        is_valid = rng.random() < 0.5
    else:
        is_valid = force_valid
    
    if is_valid:
        # Generate valid bracket sequence
        symbols = _generate_valid_brackets(length, rng)
        target = ["V"]
    else:
        # Generate invalid bracket sequence
        symbols = _generate_invalid_brackets(length, rng)
        target = ["X"]
    
    return OpticalManifestEntry(
        split=split,
        symbols=symbols,
        target_symbols=target,
        length=len(symbols),
        task="bracket",
        difficulty_tag=f"len{len(symbols)}",
        example_id=example_id,
        seed=seed,
        sequence_seed=seed,
    )


def _generate_valid_brackets(length: int, rng: random.Random) -> List[str]:
    """Generate a valid (balanced) bracket sequence."""
    if length <= 0:
        return []
    if length % 2 != 0:
        length -= 1  # Make even
    
    n = length // 2
    if n == 0:
        return []
    
    # Generate using balanced parentheses algorithm
    result: List[str] = []
    open_count = 0
    close_count = 0
    
    for _ in range(length):
        if open_count == n:
            result.append(")")
            close_count += 1
        elif close_count == open_count:
            result.append("(")
            open_count += 1
        else:
            if rng.random() < 0.5:
                result.append("(")
                open_count += 1
            else:
                result.append(")")
                close_count += 1
    
    return result


def _generate_invalid_brackets(length: int, rng: random.Random) -> List[str]:
    """Generate an invalid (unbalanced) bracket sequence."""
    if length <= 0:
        return [")"]  # Minimal invalid
    
    # Strategy: generate random sequence and ensure it's invalid
    while True:
        symbols = [rng.choice(["(", ")"]) for _ in range(length)]
        if not _is_balanced(symbols):
            return symbols
        # If accidentally valid, flip one bracket
        if symbols:
            idx = rng.randint(0, len(symbols) - 1)
            symbols[idx] = ")" if symbols[idx] == "(" else "("
            if not _is_balanced(symbols):
                return symbols


def _is_balanced(symbols: Sequence[str]) -> bool:
    """Check if bracket sequence is balanced."""
    count = 0
    for s in symbols:
        if s == "(":
            count += 1
        elif s == ")":
            count -= 1
        if count < 0:
            return False
    return count == 0


def generate_task3_mod_entry(
    split: str,
    num_digits: int,
    modulo: int,
    example_id: str,
    seed: int,
    use_ood_digits: bool = False,
) -> OpticalManifestEntry:
    """Generate a Task3 (Mod) manifest entry.
    
    Expression format: "d1 d2 ... dn % m"
    Target: (d1 + d2 + ... + dn) % m as single digit
    
    Parameters
    ----------
    split : str
        Split name
    num_digits : int
        Number of digits to sum
    modulo : int
        Modulo value (1-9)
    example_id : str
        Unique example identifier
    seed : int
        Random seed
    use_ood_digits : bool
        If True, include digit 9
    """
    rng = random.Random(seed)
    
    if use_ood_digits:
        pool = TASK3_ALL_DIGITS
        digits = [rng.choice(pool) for _ in range(num_digits - 1)]
        # Ensure at least one 9
        insert_pos = rng.randint(0, len(digits))
        digits.insert(insert_pos, "9")
    else:
        digits = [rng.choice(TASK3_IID_DIGITS) for _ in range(num_digits)]
    
    # Calculate result
    digit_sum = sum(int(d) for d in digits)
    result = digit_sum % modulo
    
    # Build expression: digits + % + modulo_digit
    symbols = digits + ["%", str(modulo)]
    target = [str(result)]
    
    return OpticalManifestEntry(
        split=split,
        symbols=symbols,
        target_symbols=target,
        length=len(symbols),
        task="mod",
        difficulty_tag=f"digits{num_digits}_mod{modulo}",
        example_id=example_id,
        seed=seed,
        sequence_seed=seed,
    )


def compute_manifest_hash(entries: List[OpticalManifestEntry]) -> str:
    """Compute SHA256 hash of manifest entries.
    
    Hash is computed on sorted JSON lines (order-independent).
    """
    lines = [e.to_json() for e in entries]
    lines.sort()  # Ensure order-independence
    content = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def generate_optical_manifest(
    task: str,
    output_path: Path,
    seed: int = 42,
    iid_train_count: int = 500,
    iid_val_count: int = 100,
    iid_test_count: int = 200,
    ood_length_count: int = 200,
    ood_symbol_count: int = 200,
    iid_length_range: tuple = (2, 8),
    ood_length_range: tuple = (10, 16),
) -> str:
    """Generate a complete manifest for a task.
    
    Returns the manifest hash.
    """
    rng = random.Random(seed)
    entries: List[OpticalManifestEntry] = []
    example_counter = 0
    
    def next_id() -> str:
        nonlocal example_counter
        example_counter += 1
        return f"{task}_{example_counter:06d}"
    
    def next_seed() -> int:
        return rng.randint(0, 2**31 - 1)
    
    if task == "mirror":
        # IID splits
        for split, count in [
            ("iid_train", iid_train_count),
            ("iid_val", iid_val_count),
            ("iid_test", iid_test_count),
        ]:
            for _ in range(count):
                length = rng.randint(*iid_length_range)
                entries.append(generate_task1_mirror_entry(
                    split=split,
                    length=length,
                    example_id=next_id(),
                    seed=next_seed(),
                    use_ood_symbols=False,
                ))
        
        # OOD-length
        for _ in range(ood_length_count):
            length = rng.randint(*ood_length_range)
            entries.append(generate_task1_mirror_entry(
                split="ood_length",
                length=length,
                example_id=next_id(),
                seed=next_seed(),
                use_ood_symbols=False,
            ))
        
        # OOD-symbol (with F)
        for _ in range(ood_symbol_count):
            length = rng.randint(*iid_length_range)
            entries.append(generate_task1_mirror_entry(
                split="ood_symbol",
                length=length,
                example_id=next_id(),
                seed=next_seed(),
                use_ood_symbols=True,
            ))
    
    elif task == "bracket":
        # IID splits
        for split, count in [
            ("iid_train", iid_train_count),
            ("iid_val", iid_val_count),
            ("iid_test", iid_test_count),
        ]:
            for _ in range(count):
                length = rng.randint(*iid_length_range) * 2  # Even length
                entries.append(generate_task2_bracket_entry(
                    split=split,
                    length=length,
                    example_id=next_id(),
                    seed=next_seed(),
                ))
        
        # OOD-length
        for _ in range(ood_length_count):
            length = rng.randint(*ood_length_range) * 2
            entries.append(generate_task2_bracket_entry(
                split="ood_length",
                length=length,
                example_id=next_id(),
                seed=next_seed(),
            ))
    
    elif task == "mod":
        # IID splits
        for split, count in [
            ("iid_train", iid_train_count),
            ("iid_val", iid_val_count),
            ("iid_test", iid_test_count),
        ]:
            for _ in range(count):
                num_digits = rng.randint(*iid_length_range)
                modulo = rng.randint(2, 8)
                entries.append(generate_task3_mod_entry(
                    split=split,
                    num_digits=num_digits,
                    modulo=modulo,
                    example_id=next_id(),
                    seed=next_seed(),
                    use_ood_digits=False,
                ))
        
        # OOD-length
        for _ in range(ood_length_count):
            num_digits = rng.randint(*ood_length_range)
            modulo = rng.randint(2, 8)
            entries.append(generate_task3_mod_entry(
                split="ood_length",
                num_digits=num_digits,
                modulo=modulo,
                example_id=next_id(),
                seed=next_seed(),
                use_ood_digits=False,
            ))
        
        # OOD-symbol (with 9)
        for _ in range(ood_symbol_count):
            num_digits = rng.randint(*iid_length_range)
            modulo = rng.randint(2, 8)
            entries.append(generate_task3_mod_entry(
                split="ood_symbol",
                num_digits=num_digits,
                modulo=modulo,
                example_id=next_id(),
                seed=next_seed(),
                use_ood_digits=True,
            ))
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Shuffle within splits for training
    rng.shuffle(entries)
    
    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.to_json() + "\n")
    
    # Return hash
    return compute_manifest_hash(entries)


def main():
    parser = argparse.ArgumentParser(description="Generate optical domain manifests")
    parser.add_argument("--task", required=True, choices=["mirror", "bracket", "mod"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iid-train", type=int, default=500)
    parser.add_argument("--iid-val", type=int, default=100)
    parser.add_argument("--iid-test", type=int, default=200)
    parser.add_argument("--ood-length", type=int, default=200)
    parser.add_argument("--ood-symbol", type=int, default=200)
    
    args = parser.parse_args()
    
    manifest_hash = generate_optical_manifest(
        task=args.task,
        output_path=args.output,
        seed=args.seed,
        iid_train_count=args.iid_train,
        iid_val_count=args.iid_val,
        iid_test_count=args.iid_test,
        ood_length_count=args.ood_length,
        ood_symbol_count=args.ood_symbol,
    )
    
    print(f"Generated manifest: {args.output}")
    print(f"Manifest hash: {manifest_hash}")


if __name__ == "__main__":
    main()

