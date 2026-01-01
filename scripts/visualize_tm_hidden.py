#!/usr/bin/env python3
"""
Visualize hidden states from Turing Machine experiments.

Generates plots for:
1. Hidden state trajectories (PCA)
2. FFT of hidden state time series (internal clock detection)
3. Phase-locking analysis (symbol boundary alignment)
4. Layer-wise norm evolution
"""

import argparse
import pickle
import sys
import io
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_hidden_states(pkl_path: Path):
    """Load hidden states from pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_hidden_trajectory(hidden_states, symbols, output_path: Path):
    """
    Plot PCA of hidden state trajectory.
    
    This shows how the model's internal state evolves as it processes
    each symbol. If there's an "internal clock", we expect to see
    regular/cyclic patterns.
    """
    # hidden_states: (seq_len, d_model)
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    trajectory = pca.fit_transform(hidden_states)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: 2D trajectory with time color
    ax1 = axes[0]
    scatter = ax1.scatter(
        trajectory[:, 0], trajectory[:, 1],
        c=np.arange(len(trajectory)), cmap='viridis',
        s=50, alpha=0.7
    )
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'k-', alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Time Step')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_title('Hidden State Trajectory (PCA)')
    
    # Annotate with symbols
    for i, sym in enumerate(symbols[:len(trajectory)]):
        ax1.annotate(sym, (trajectory[i, 0], trajectory[i, 1]), fontsize=8)
    
    # Right: Component evolution over time
    ax2 = axes[1]
    ax2.plot(trajectory[:, 0], label='PC1', linewidth=2)
    ax2.plot(trajectory[:, 1], label='PC2', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Component Value')
    ax2.set_title('Principal Components Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return pca.explained_variance_ratio_


def plot_fft_analysis(hidden_states, sample_rate: int, output_path: Path):
    """
    FFT analysis of hidden state time series.
    
    If there's an internal clock, we expect to see peaks at frequencies
    corresponding to symbol duration.
    """
    # Average across hidden dimensions
    signal = hidden_states.mean(axis=1)  # (seq_len,)
    
    # Compute FFT
    fft_result = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/sample_rate)
    magnitudes = np.abs(fft_result)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: Time domain
    ax1 = axes[0]
    ax1.plot(signal, linewidth=1)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Mean Hidden State')
    ax1.set_title('Hidden State Signal (Mean Across Dimensions)')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Frequency domain
    ax2 = axes[1]
    ax2.plot(freqs[1:], magnitudes[1:], linewidth=1)  # Skip DC component
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('FFT of Hidden State Signal')
    ax2.grid(True, alpha=0.3)
    
    # Find dominant frequency
    dominant_idx = np.argmax(magnitudes[1:]) + 1
    dominant_freq = freqs[dominant_idx]
    ax2.axvline(dominant_freq, color='r', linestyle='--', 
                label=f'Dominant: {dominant_freq:.2f} Hz')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return dominant_freq


def plot_layer_norms(hidden_states, symbols, output_path: Path):
    """
    Plot hidden state norms over time.
    
    This shows how the model's internal "energy" evolves.
    """
    # Compute L2 norm at each time step
    norms = np.linalg.norm(hidden_states, axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(norms, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('L2 Norm')
    ax.set_title('Hidden State Norm Evolution')
    ax.grid(True, alpha=0.3)
    
    # Mark symbol boundaries (assuming ~8 frames per symbol based on encoding)
    # This is approximate; adjust based on actual encoding
    frames_per_symbol = 8  # Rough estimate
    for i, sym in enumerate(symbols):
        x = i * frames_per_symbol
        if x < len(norms):
            ax.axvline(x, color='gray', alpha=0.3, linestyle=':')
            ax.annotate(sym, (x, norms[min(x, len(norms)-1)]), fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return norms


def plot_phase_locking(hidden_states, symbols, output_path: Path):
    """
    Analyze phase-locking between hidden states and symbol boundaries.
    
    If the model has an internal clock, hidden states should show
    consistent patterns at symbol boundaries.
    """
    # Estimate frames per symbol
    frames_per_symbol = len(hidden_states) // len(symbols)
    if frames_per_symbol < 1:
        frames_per_symbol = 1
    
    # Extract hidden states at symbol boundaries
    boundary_states = []
    for i in range(len(symbols)):
        idx = i * frames_per_symbol
        if idx < len(hidden_states):
            boundary_states.append(hidden_states[idx])
    
    boundary_states = np.array(boundary_states)
    
    # PCA on boundary states
    if len(boundary_states) > 2:
        pca = PCA(n_components=2)
        boundary_2d = pca.fit_transform(boundary_states)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Color by symbol type
        unique_symbols = list(set(symbols))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_symbols)))
        symbol_to_color = {s: c for s, c in zip(unique_symbols, colors)}
        
        for i, sym in enumerate(symbols[:len(boundary_2d)]):
            ax.scatter(boundary_2d[i, 0], boundary_2d[i, 1],
                      c=[symbol_to_color[sym]], s=100, label=sym if i == symbols.index(sym) else '')
            ax.annotate(f'{i}:{sym}', (boundary_2d[i, 0], boundary_2d[i, 1]), fontsize=8)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Hidden States at Symbol Boundaries')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return pca.explained_variance_ratio_
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize Turing Machine Hidden States")
    parser.add_argument("--run-dir", type=Path, required=True,
                       help="Directory containing hidden_states.pkl")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory for plots (default: run_dir/viz)")
    parser.add_argument("--sample-idx", type=int, default=0,
                       help="Sample index to visualize")
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = args.output_dir or (args.run_dir / "viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load hidden states
    pkl_path = args.run_dir / "hidden_states.pkl"
    if not pkl_path.exists():
        print(f"Error: {pkl_path} not found")
        return
    
    print(f"Loading hidden states from {pkl_path}...")
    data = load_hidden_states(pkl_path)
    
    print(f"Found {len(data)} samples")
    
    # Select sample
    sample = data[args.sample_idx]
    print(f"\nAnalyzing sample {args.sample_idx}:")
    print(f"  Input: {sample['input_symbols']}")
    print(f"  Output: {sample['output_symbols']}")
    print(f"  Prediction: {sample['prediction']}")
    
    # Get final layer hidden states
    hidden_states = sample['hidden_states']
    if isinstance(hidden_states, list):
        hidden_states = hidden_states[-1] if hidden_states else None
    
    if hidden_states is None:
        print("Error: No hidden states found")
        return
    
    print(f"  Hidden states shape: {hidden_states.shape}")
    
    symbols = sample['input_symbols']
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # 1. PCA trajectory
    print("\n1. Hidden State Trajectory (PCA)...")
    variance_ratio = plot_hidden_trajectory(
        hidden_states, symbols, output_dir / "trajectory.png"
    )
    print(f"   Variance explained: PC1={variance_ratio[0]:.1%}, PC2={variance_ratio[1]:.1%}")
    
    # 2. FFT analysis
    print("\n2. FFT Analysis (Internal Clock Detection)...")
    dominant_freq = plot_fft_analysis(
        hidden_states, 100, output_dir / "fft.png"
    )
    print(f"   Dominant frequency: {dominant_freq:.2f} Hz")
    
    # 3. Norm evolution
    print("\n3. Hidden State Norm Evolution...")
    norms = plot_layer_norms(
        hidden_states, symbols, output_dir / "norms.png"
    )
    print(f"   Norm range: [{norms.min():.2f}, {norms.max():.2f}]")
    print(f"   Norm growth rate: {(norms[-1] - norms[0]) / len(norms):.4f} per step")
    
    # 4. Phase-locking analysis
    print("\n4. Phase-Locking Analysis...")
    phase_ratio = plot_phase_locking(
        hidden_states, symbols, output_dir / "phase_locking.png"
    )
    if phase_ratio is not None:
        print(f"   Boundary variance: PC1={phase_ratio[0]:.1%}, PC2={phase_ratio[1]:.1%}")
    
    print(f"\nVisualizations saved to {output_dir}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Internal Clock Analysis Summary")
    print("=" * 60)
    print(f"Dominant FFT frequency: {dominant_freq:.2f} Hz")
    print(f"Expected symbol frequency: ~10 Hz (0.1s per symbol)")
    print(f"Ratio: {dominant_freq / 10:.2f}x")
    
    if 0.8 <= dominant_freq / 10 <= 1.2:
        print("\n[EVIDENCE] Dominant frequency matches symbol rate!")
        print("This suggests the model may have developed an internal clock.")
    else:
        print("\n[NOTE] Dominant frequency does not match symbol rate.")
        print("Internal clock evidence is weak or absent.")


if __name__ == "__main__":
    main()

