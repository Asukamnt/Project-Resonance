#!/usr/bin/env python3
"""
Turing Machine Simulation Tasks for Waveform Reasoning

Level 1: Counter - Count occurrences of a symbol
Level 2: Parity - XOR of binary sequence
Level 3: Bracket Depth - Maximum nesting depth
Level 4: Binary Addition - Multi-step carry propagation
Level 5: Program Execution - Simple instruction sequence

Each level tests progressively more complex computational abilities.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
import sys

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def generate_counter_samples(
    num_samples: int,
    min_count: int = 1,
    max_count: int = 15,
    symbols: List[str] = ['A', 'B', 'C'],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Level 1: Counter Task
    Input: A sequence of identical symbols (e.g., AAAAA)
    Output: The count as digits (e.g., 5)
    
    This tests: state accumulation, counting ability
    """
    random.seed(seed)
    samples = []
    
    for i in range(num_samples):
        count = random.randint(min_count, max_count)
        symbol = random.choice(symbols)
        
        input_symbols = [symbol] * count
        output_symbols = list(str(count))  # "15" -> ["1", "5"]
        
        samples.append({
            "id": f"counter_{i:04d}",
            "task": "counter",
            "level": 1,
            "input_symbols": input_symbols,
            "output_symbols": output_symbols,
            "metadata": {
                "count": count,
                "symbol": symbol
            }
        })
    
    return samples


def generate_parity_samples(
    num_samples: int,
    min_length: int = 3,
    max_length: int = 15,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Level 2: Parity Task (XOR)
    Input: A binary sequence (e.g., 1 0 1 1 0)
    Output: 1 if odd number of 1s, 0 if even
    
    This tests: XOR logic, state toggle
    """
    random.seed(seed)
    samples = []
    
    for i in range(num_samples):
        length = random.randint(min_length, max_length)
        bits = [random.choice(['0', '1']) for _ in range(length)]
        
        parity = sum(1 for b in bits if b == '1') % 2
        output_symbols = [str(parity)]
        
        samples.append({
            "id": f"parity_{i:04d}",
            "task": "parity",
            "level": 2,
            "input_symbols": bits,
            "output_symbols": output_symbols,
            "metadata": {
                "length": length,
                "num_ones": sum(1 for b in bits if b == '1'),
                "parity": parity
            }
        })
    
    return samples


def generate_bracket_depth_samples(
    num_samples: int,
    min_depth: int = 1,
    max_depth: int = 8,
    min_length: int = 4,
    max_length: int = 20,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Level 3: Bracket Depth Task
    Input: A balanced bracket sequence (e.g., ( ( ) ( ( ) ) ))
    Output: Maximum nesting depth (e.g., 3)
    
    This tests: stack simulation, depth tracking
    """
    random.seed(seed)
    samples = []
    
    def generate_balanced_brackets(target_depth: int, target_length: int) -> Tuple[List[str], int]:
        """Generate a balanced bracket sequence with specified max depth."""
        result = []
        current_depth = 0
        max_reached = 0
        open_count = 0
        
        while len(result) < target_length:
            remaining = target_length - len(result)
            
            # Must close all open brackets
            if remaining <= open_count:
                result.append(')')
                current_depth -= 1
                open_count -= 1
            # Can open if not at target depth and have room
            elif current_depth < target_depth and remaining > open_count + 1:
                if random.random() < 0.6:  # Bias towards opening
                    result.append('(')
                    current_depth += 1
                    open_count += 1
                    max_reached = max(max_reached, current_depth)
                elif open_count > 0:
                    result.append(')')
                    current_depth -= 1
                    open_count -= 1
                else:
                    result.append('(')
                    current_depth += 1
                    open_count += 1
                    max_reached = max(max_reached, current_depth)
            elif open_count > 0:
                result.append(')')
                current_depth -= 1
                open_count -= 1
            else:
                result.append('(')
                current_depth += 1
                open_count += 1
                max_reached = max(max_reached, current_depth)
        
        return result, max_reached
    
    for i in range(num_samples):
        target_depth = random.randint(min_depth, max_depth)
        # Length must be even for balanced brackets
        length = random.randint(min_length, max_length)
        if length % 2 != 0:
            length += 1
        
        brackets, actual_depth = generate_balanced_brackets(target_depth, length)
        output_symbols = list(str(actual_depth))
        
        samples.append({
            "id": f"depth_{i:04d}",
            "task": "bracket_depth",
            "level": 3,
            "input_symbols": brackets,
            "output_symbols": output_symbols,
            "metadata": {
                "length": len(brackets),
                "max_depth": actual_depth,
                "target_depth": target_depth
            }
        })
    
    return samples


def generate_binary_addition_samples(
    num_samples: int,
    min_bits: int = 2,
    max_bits: int = 4,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Level 4: Binary Addition Task
    Input: Two binary numbers separated by '+' (e.g., 101 + 011)
    Output: Binary sum (e.g., 1000)
    
    This tests: multi-step carry propagation, arithmetic chain
    """
    random.seed(seed)
    samples = []
    
    for i in range(num_samples):
        bits = random.randint(min_bits, max_bits)
        
        # Generate two random binary numbers
        a = random.randint(0, 2**bits - 1)
        b = random.randint(0, 2**bits - 1)
        result = a + b
        
        # Convert to binary strings (without '0b' prefix)
        a_bin = format(a, f'0{bits}b')
        b_bin = format(b, f'0{bits}b')
        result_bin = format(result, 'b')  # Variable length for result
        
        # Input: a + b as symbol sequence
        input_symbols = list(a_bin) + ['+'] + list(b_bin)
        output_symbols = list(result_bin)
        
        samples.append({
            "id": f"binadd_{i:04d}",
            "task": "binary_addition",
            "level": 4,
            "input_symbols": input_symbols,
            "output_symbols": output_symbols,
            "metadata": {
                "a": a,
                "b": b,
                "result": result,
                "bits": bits,
                "has_carry": result >= 2**bits
            }
        })
    
    return samples


def generate_program_execution_samples(
    num_samples: int,
    min_steps: int = 2,
    max_steps: int = 5,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Level 5: Program Execution Task
    Input: Initial value + instruction sequence
    Output: Final value after execution
    
    Instructions:
    - I: Increment (+1)
    - D: Decrement (-1)
    - T: Triple (*3)
    - H: Halve (//2)
    
    Example: 3 I I D → 3+1+1-1 = 4
    
    This tests: instruction interpretation, multi-step execution
    """
    random.seed(seed)
    samples = []
    
    instructions = ['I', 'D', 'T', 'H']
    
    for i in range(num_samples):
        # Initial value (keep small to avoid overflow)
        initial = random.randint(1, 9)
        
        # Generate instruction sequence
        num_steps = random.randint(min_steps, max_steps)
        program = [random.choice(instructions) for _ in range(num_steps)]
        
        # Execute program
        value = initial
        for inst in program:
            if inst == 'I':
                value += 1
            elif inst == 'D':
                value = max(0, value - 1)  # Clamp at 0
            elif inst == 'T':
                value *= 3
            elif inst == 'H':
                value //= 2
        
        # Clamp result to reasonable range
        value = min(value, 999)
        
        input_symbols = [str(initial)] + program
        output_symbols = list(str(value))
        
        samples.append({
            "id": f"prog_{i:04d}",
            "task": "program_execution",
            "level": 5,
            "input_symbols": input_symbols,
            "output_symbols": output_symbols,
            "metadata": {
                "initial": initial,
                "program": program,
                "result": value,
                "num_steps": num_steps
            }
        })
    
    return samples


def create_manifests(
    output_dir: Path,
    level: int,
    train_size: int = 500,
    val_size: int = 50,
    test_size: int = 100,
    ood_size: int = 100,
    seed: int = 42
) -> Dict[str, Path]:
    """Create train/val/test/ood manifests for a given level."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Level-specific generators and OOD configs
    if level == 1:  # Counter (simplified: 1-9 only, single digit output)
        generator = generate_counter_samples
        train_kwargs = {"min_count": 1, "max_count": 7}  # Train on 1-7
        ood_kwargs = {"min_count": 8, "max_count": 9}  # OOD: 8-9
    elif level == 2:  # Parity
        generator = generate_parity_samples
        train_kwargs = {"min_length": 3, "max_length": 10}
        ood_kwargs = {"min_length": 11, "max_length": 20}  # OOD: longer sequences
    elif level == 3:  # Bracket Depth
        generator = generate_bracket_depth_samples
        train_kwargs = {"min_depth": 1, "max_depth": 5, "min_length": 4, "max_length": 16}
        ood_kwargs = {"min_depth": 6, "max_depth": 10, "min_length": 16, "max_length": 32}
    elif level == 4:  # Binary Addition
        generator = generate_binary_addition_samples
        train_kwargs = {"min_bits": 2, "max_bits": 4}
        ood_kwargs = {"min_bits": 5, "max_bits": 6}  # OOD: more bits
    elif level == 5:  # Program Execution
        generator = generate_program_execution_samples
        train_kwargs = {"min_steps": 2, "max_steps": 4}
        ood_kwargs = {"min_steps": 5, "max_steps": 8}  # OOD: longer programs
    else:
        raise ValueError(f"Unknown level: {level}")
    
    manifests = {}
    
    # Generate splits with different seeds
    for split, size, kwargs, split_seed in [
        ("train", train_size, train_kwargs, seed),
        ("val", val_size, train_kwargs, seed + 1),
        ("iid_test", test_size, train_kwargs, seed + 2),
        ("ood_test", ood_size, ood_kwargs, seed + 3),
    ]:
        samples = generator(num_samples=size, seed=split_seed, **kwargs)
        
        # Add split info
        for sample in samples:
            sample["split"] = split
        
        # Save manifest
        manifest_path = output_dir / f"level{level}_{split}.jsonl"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        manifests[split] = manifest_path
        print(f"  [Level {level}] {split}: {len(samples)} samples -> {manifest_path}")
    
    return manifests


def main():
    parser = argparse.ArgumentParser(description="Generate Turing Machine Task Manifests")
    parser.add_argument("--levels", type=str, default="1,2,3,4,5", 
                        help="Comma-separated list of levels to generate")
    parser.add_argument("--output-dir", type=Path, default=Path("manifests/turing_machine"))
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--ood-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    levels = [int(l.strip()) for l in args.levels.split(',')]
    
    print("=" * 60)
    print("Turing Machine Task Data Generation")
    print("=" * 60)
    
    all_manifests = {}
    for level in levels:
        print(f"\nGenerating Level {level}...")
        manifests = create_manifests(
            output_dir=args.output_dir,
            level=level,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            ood_size=args.ood_size,
            seed=args.seed
        )
        all_manifests[level] = manifests
    
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    
    # Summary
    print("\nLevel Descriptions:")
    print("  Level 1: Counter       - Count symbol occurrences")
    print("  Level 2: Parity        - XOR of binary sequence")
    print("  Level 3: Bracket Depth - Max nesting depth")
    print("  Level 4: Binary Add    - Multi-bit addition with carry")
    print("  Level 5: Program Exec  - Instruction sequence execution")


if __name__ == "__main__":
    main()

