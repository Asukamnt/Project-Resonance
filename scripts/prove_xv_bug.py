"""Minimal proof of the X→V frequency bug and its fix.

This script demonstrates:
1. OLD method (tile): X template repeated → decoded as V (BUG!)
2. NEW method (continuous): X continuous wave → decoded as X (CORRECT)

Output: 2x2 confusion matrices showing the fix.
"""
import numpy as np
from jericho.symbols import SR, encode_symbols_to_wave
from jericho.scorer import decode_wave_to_symbols

FRAME_SIZE = 160
NUM_FRAMES = 10  # 10 frames = 1600 samples = 100ms

def old_method_tile(symbol: str) -> np.ndarray:
    """OLD: Tile 160-sample template 10 times."""
    wave = encode_symbols_to_wave([symbol], fixed_phase=0.0)
    template = wave[:FRAME_SIZE]
    return np.tile(template, NUM_FRAMES)

def new_method_continuous(symbol: str) -> np.ndarray:
    """NEW: Generate continuous 1600-sample waveform."""
    freq = 1900.0 if symbol == 'V' else 1950.0
    t = np.arange(NUM_FRAMES * FRAME_SIZE) / SR
    return (0.8 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def main():
    print("=" * 60)
    print("MINIMAL PROOF: X→V Frequency Bug")
    print("=" * 60)
    
    # Test both methods
    results = {"old": {}, "new": {}}
    
    for symbol in ['V', 'X']:
        # OLD method
        wave_old = old_method_tile(symbol)
        decoded_old = decode_wave_to_symbols(wave_old)
        results["old"][symbol] = decoded_old[0] if decoded_old else "?"
        
        # NEW method
        wave_new = new_method_continuous(symbol)
        decoded_new = decode_wave_to_symbols(wave_new)
        results["new"][symbol] = decoded_new[0] if decoded_new else "?"
    
    print("\n[OLD METHOD - Tile 160 samples x 10]")
    print(f"  Input V → Decoded: {results['old']['V']}")
    print(f"  Input X → Decoded: {results['old']['X']} {'← BUG!' if results['old']['X'] != 'X' else ''}")
    
    print("\n[NEW METHOD - Continuous 1600 samples]")
    print(f"  Input V → Decoded: {results['new']['V']}")
    print(f"  Input X → Decoded: {results['new']['X']}")
    
    # Confusion matrices
    print("\n" + "=" * 60)
    print("CONFUSION MATRICES")
    print("=" * 60)
    
    print("\nOLD (Tile) - BROKEN:")
    print("           Predicted")
    print("           V    X")
    print(f"  True V   {'1' if results['old']['V']=='V' else '0'}    {'1' if results['old']['V']=='X' else '0'}")
    print(f"  True X   {'1' if results['old']['X']=='V' else '0'}    {'1' if results['old']['X']=='X' else '0'}")
    
    print("\nNEW (Continuous) - FIXED:")
    print("           Predicted")
    print("           V    X")
    print(f"  True V   {'1' if results['new']['V']=='V' else '0'}    {'1' if results['new']['V']=='X' else '0'}")
    print(f"  True X   {'1' if results['new']['X']=='V' else '0'}    {'1' if results['new']['X']=='X' else '0'}")
    
    # FFT analysis
    print("\n" + "=" * 60)
    print("FFT PEAK FREQUENCIES")
    print("=" * 60)
    
    freqs = np.fft.rfftfreq(1600, d=1.0/SR)
    for method_name, method_fn in [("OLD", old_method_tile), ("NEW", new_method_continuous)]:
        print(f"\n{method_name} method:")
        for symbol in ['V', 'X']:
            wave = method_fn(symbol)
            spectrum = np.abs(np.fft.rfft(wave))
            spectrum[0] = 0  # ignore DC
            peak_freq = freqs[np.argmax(spectrum)]
            expected = 1900.0 if symbol == 'V' else 1950.0
            status = "[OK]" if abs(peak_freq - expected) < 20 else "[WRONG!]"
            print(f"  {symbol}: expected={expected:.0f}Hz, got={peak_freq:.0f}Hz {status}")

if __name__ == "__main__":
    main()

