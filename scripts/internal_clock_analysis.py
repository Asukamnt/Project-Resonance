#!/usr/bin/env python3
"""
Deep analysis of internal clock hypothesis.

Analyzes:
1. FFT statistics across multiple samples
2. Phase-locking to symbol boundaries
3. Comparison across different tasks
"""

import argparse
import pickle
import sys
import io
from pathlib import Path
from typing import List, Dict, Any

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_all_hidden_states(run_dir: Path) -> List[Dict[str, Any]]:
    """Load all hidden states from a run directory."""
    pkl_path = run_dir / "hidden_states.pkl"
    if not pkl_path.exists():
        return []
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_fft_dominant_freq(hidden_states: np.ndarray, sample_rate: float = 100.0) -> float:
    """Compute dominant frequency from hidden states."""
    if hidden_states.ndim == 1:
        signal = hidden_states
    else:
        signal = hidden_states.mean(axis=1)
    
    fft_result = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/sample_rate)
    magnitudes = np.abs(fft_result)
    
    # Skip DC component
    if len(magnitudes) > 1:
        dominant_idx = np.argmax(magnitudes[1:]) + 1
        return freqs[dominant_idx]
    return 0.0


def compute_oscillation_period(hidden_states: np.ndarray) -> float:
    """Estimate oscillation period from autocorrelation."""
    if hidden_states.ndim == 1:
        signal = hidden_states
    else:
        signal = hidden_states.mean(axis=1)
    
    # Normalize
    signal = signal - signal.mean()
    
    # Autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    
    # Find first peak after lag 0
    peaks = []
    for i in range(2, len(autocorr) - 1):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            peaks.append(i)
            break
    
    if peaks:
        return peaks[0]
    return 0.0


def compute_norm_stability(hidden_states: np.ndarray) -> Dict[str, float]:
    """Compute norm statistics."""
    norms = np.linalg.norm(hidden_states, axis=-1) if hidden_states.ndim > 1 else np.abs(hidden_states)
    
    return {
        'mean': float(norms.mean()),
        'std': float(norms.std()),
        'cv': float(norms.std() / norms.mean()) if norms.mean() > 0 else 0,  # Coefficient of variation
        'range': float(norms.max() - norms.min()),
    }


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    """Analyze a single run for internal clock evidence."""
    data = load_all_hidden_states(run_dir)
    if not data:
        return {}
    
    results = {
        'num_samples': len(data),
        'dominant_freqs': [],
        'oscillation_periods': [],
        'norm_stats': [],
    }
    
    for sample in data:
        hs = sample['hidden_states']
        if hs is None or (isinstance(hs, np.ndarray) and hs.size == 0):
            continue
        if isinstance(hs, list):
            hs = hs[-1] if hs else None
        if hs is None or hs.ndim < 2:
            continue
        
        # FFT analysis
        freq = compute_fft_dominant_freq(hs)
        results['dominant_freqs'].append(freq)
        
        # Oscillation period
        period = compute_oscillation_period(hs)
        results['oscillation_periods'].append(period)
        
        # Norm stability
        norm_stats = compute_norm_stability(hs)
        results['norm_stats'].append(norm_stats)
    
    # Aggregate statistics
    if results['dominant_freqs']:
        results['freq_mean'] = float(np.mean(results['dominant_freqs']))
        results['freq_std'] = float(np.std(results['dominant_freqs']))
    
    if results['oscillation_periods']:
        results['period_mean'] = float(np.mean(results['oscillation_periods']))
        results['period_std'] = float(np.std(results['oscillation_periods']))
    
    if results['norm_stats']:
        results['norm_cv_mean'] = float(np.mean([s['cv'] for s in results['norm_stats']]))
    
    return results


def compare_tasks(run_dirs: Dict[str, Path], output_dir: Path):
    """Compare internal clock evidence across different tasks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    for task_name, run_dir in run_dirs.items():
        print(f"\nAnalyzing {task_name}...")
        results = analyze_run(run_dir)
        if results:
            all_results[task_name] = results
            print(f"  Samples: {results['num_samples']}")
            if 'freq_mean' in results:
                print(f"  Dominant freq: {results['freq_mean']:.2f} +/- {results['freq_std']:.2f} Hz")
            if 'period_mean' in results:
                print(f"  Oscillation period: {results['period_mean']:.1f} +/- {results['period_std']:.1f} steps")
            if 'norm_cv_mean' in results:
                print(f"  Norm CV (stability): {results['norm_cv_mean']:.4f}")
    
    # Plot comparison
    if len(all_results) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        tasks = list(all_results.keys())
        
        # Dominant frequency
        ax1 = axes[0]
        freqs = [all_results[t].get('freq_mean', 0) for t in tasks]
        freq_stds = [all_results[t].get('freq_std', 0) for t in tasks]
        ax1.bar(tasks, freqs, yerr=freq_stds, capsize=5)
        ax1.axhline(10, color='r', linestyle='--', label='Expected symbol rate (10 Hz)')
        ax1.set_ylabel('Dominant Frequency (Hz)')
        ax1.set_title('FFT Dominant Frequency by Task')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Oscillation period
        ax2 = axes[1]
        periods = [all_results[t].get('period_mean', 0) for t in tasks]
        period_stds = [all_results[t].get('period_std', 0) for t in tasks]
        ax2.bar(tasks, periods, yerr=period_stds, capsize=5)
        ax2.axhline(8, color='r', linestyle='--', label='Expected frames/symbol (8)')
        ax2.set_ylabel('Oscillation Period (steps)')
        ax2.set_title('Autocorrelation Period by Task')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # Norm stability (lower CV = more stable)
        ax3 = axes[2]
        cvs = [all_results[t].get('norm_cv_mean', 0) for t in tasks]
        ax3.bar(tasks, cvs)
        ax3.set_ylabel('Norm Coefficient of Variation')
        ax3.set_title('Hidden State Stability by Task')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'task_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved comparison plot to {output_dir / 'task_comparison.png'}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Internal Clock Analysis")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/turing_machine"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/internal_clock"))
    args = parser.parse_args()
    
    print("=" * 60)
    print("Internal Clock Hypothesis Analysis")
    print("=" * 60)
    
    # Find all run directories
    run_dirs = {}
    for level in range(1, 6):
        pattern = f"level{level}_*"
        matches = list(args.runs_dir.glob(pattern))
        if matches:
            # Use most recent run
            latest = sorted(matches)[-1]
            run_dirs[f"Level {level}"] = latest
    
    if not run_dirs:
        print("No run directories found!")
        return
    
    print(f"\nFound {len(run_dirs)} task runs:")
    for name, path in run_dirs.items():
        print(f"  {name}: {path.name}")
    
    # Compare tasks
    results = compare_tasks(run_dirs, args.output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("Internal Clock Hypothesis Summary")
    print("=" * 60)
    
    expected_freq = 10.0  # Hz
    expected_period = 8.0  # steps
    
    clock_evidence = []
    for task, r in results.items():
        freq = r.get('freq_mean', 0)
        period = r.get('period_mean', 0)
        
        freq_ratio = freq / expected_freq if expected_freq > 0 else 0
        period_ratio = period / expected_period if expected_period > 0 else 0
        
        # Evidence score: how close to expected?
        freq_match = 1.0 - min(abs(freq_ratio - 1.0), 0.5) * 2  # 0.5-1.5x = positive evidence
        period_match = 1.0 - min(abs(period_ratio - 1.0), 0.5) * 2
        
        evidence = (freq_match + period_match) / 2
        clock_evidence.append((task, evidence, freq, period))
    
    print("\nEvidence Scores (0 = no evidence, 1 = strong evidence):")
    for task, evidence, freq, period in sorted(clock_evidence, key=lambda x: -x[1]):
        status = "STRONG" if evidence > 0.7 else "MODERATE" if evidence > 0.4 else "WEAK"
        print(f"  {task}: {evidence:.2f} [{status}]")
        print(f"    Freq: {freq:.1f} Hz (expected ~10 Hz)")
        print(f"    Period: {period:.1f} steps (expected ~8 steps)")
    
    # Overall conclusion
    avg_evidence = np.mean([e for _, e, _, _ in clock_evidence])
    print(f"\nOverall Evidence Score: {avg_evidence:.2f}")
    
    if avg_evidence > 0.6:
        print("\n[CONCLUSION] Strong evidence for internal clock formation!")
        print("The model appears to develop periodic representations aligned with symbol rate.")
    elif avg_evidence > 0.3:
        print("\n[CONCLUSION] Moderate evidence for internal clock.")
        print("Some periodic patterns detected, but not consistently aligned with symbol rate.")
    else:
        print("\n[CONCLUSION] Weak evidence for internal clock.")
        print("No consistent periodic patterns detected.")


if __name__ == "__main__":
    main()

