================================================================================
Jericho: Reasoning is Resonance
Supplementary Material Index
================================================================================

This document describes the supplementary material accompanying the paper.

--------------------------------------------------------------------------------
SUPPLEMENTARY FIGURES (figures/supp/)
--------------------------------------------------------------------------------

Fig. B1 (sup_fig_B1_temporal_norm.png)
    - Temporal norm evolution across sequence positions
    - Shows state magnitude dynamics during reasoning

Fig. B2 (sup_fig_B2_layer_norms.png)
    - Layer-wise norm comparison across model depth
    - Reveals hierarchical feature extraction patterns

Fig. B3 (sup_fig_B3_trajectory_detail.png)
    - Detailed trajectory overlays with confidence bands
    - High-resolution view of IID/OOD divergence

Fig. C1 (sup_fig_C1_tasks_overlay.png)
    - Multi-task trajectory overlay (Mirror, Bracket, Mod)
    - Shows task-specific vs. shared representation structure

Fig. C2 (sup_fig_C2_tasks_trajectories.png)
    - Per-task trajectory decomposition
    - Individual task reasoning patterns

Fig. D1 (sup_fig_D1_cross_domain_2d.png)
    - 2D projection of cross-domain hidden states
    - Audio/IPD/RF representation alignment

Fig. E1 (sup_fig_E1_attention.png)
    - Attention weight visualization
    - Cross-position alignment patterns

Fig. E2 (sup_fig_E2_state_dynamics.png)
    - SSM state dynamics across time
    - Selective state space evolution

Fig. E3 (sup_fig_E3_pruning.png)
    - State pruning ablation results
    - Optimal pruning ratio by sequence length

--------------------------------------------------------------------------------
SUPPLEMENTARY VIDEOS (figures/supp/)
--------------------------------------------------------------------------------

Video S1 (video_S1_ood_collapse.gif)
    - Duration: ~10 seconds (looped)
    - Content: OOD length generalization collapse
    - Shows: Hidden state trajectories drifting into unexplored regions
             as sequence length exceeds training distribution

Video S2 (video_S2_multi_task.gif)
    - Duration: ~10 seconds (looped)
    - Content: Multi-task reasoning dynamics
    - Shows: How the same architecture develops different trajectory
             patterns for Mirror, Bracket, and Modular arithmetic

Video S3 (video_S3_cross_domain.gif)
    - Duration: ~8 seconds (looped)
    - Content: Cross-domain transfer visualization
    - Shows: Representation alignment across Audio, IPD, and RF domains
             during the transfer learning process

Video S4 (video_S4_thought_trajectories.gif)
    - Duration: ~12 seconds (looped)
    - Content: "Trajectories of Thought" visualization
    - Shows: Full reasoning process from input encoding through
             intermediate states to output production

--------------------------------------------------------------------------------
FILE FORMAT NOTES
--------------------------------------------------------------------------------

- All static figures: PNG format, 300 DPI
- All videos: GIF format, 15 FPS, < 6 MB each
- Recommended viewer: Any modern web browser or PDF viewer with animation support

--------------------------------------------------------------------------------
CORRESPONDENCE
--------------------------------------------------------------------------------

For questions about the supplementary material, please contact the authors.

================================================================================

