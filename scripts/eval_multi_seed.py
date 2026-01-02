#!/usr/bin/env python3
"""Evaluate multiple seeds for statistical reporting."""

import torch
import json
import numpy as np
from pathlib import Path
from src.evaluation.evaluator import evaluate_single_task


def main():
    seeds = [42, 123, 456]
    results = []
    
    for seed in seeds:
        ckpt = Path(f'runs/ood_length_decay/mini_jmamba_seed{seed}/mod_seed{seed}_epoch200.pt')
        if ckpt.exists():
            print(f'Evaluating seed {seed}...')
            result = evaluate_single_task('mod', 'iid_test', checkpoint=str(ckpt), limit=100)
            results.append({'seed': seed, 'em': result['em']})
            print(f'  EM: {result["em"]:.1%}')
        else:
            print(f'Checkpoint not found: {ckpt}')
    
    if results:
        ems = [r['em'] for r in results]
        mean_em = np.mean(ems)
        std_em = np.std(ems, ddof=1)
        n = len(results)
        ci_half = 1.96 * std_em / np.sqrt(n)
        
        print(f'\nSummary (n={n}):')
        print(f'  Mean EM: {mean_em:.1%} ± {std_em:.1%}')
        print(f'  95% CI: [{mean_em - ci_half:.1%}, {mean_em + ci_half:.1%}]')
        
        # Save results
        output = {
            'task': 'mod',
            'split': 'iid_test',
            'n_seeds': n,
            'seeds': seeds[:n],
            'individual_em': ems,
            'mean_em': float(mean_em),
            'std_em': float(std_em),
            'ci_95_lower': float(mean_em - ci_half),
            'ci_95_upper': float(mean_em + ci_half),
        }
        
        with open('reports/multi_seed_task3.json', 'w') as f:
            json.dump(output, f, indent=2)
        print(f'\nSaved to reports/multi_seed_task3.json')


if __name__ == '__main__':
    main()


