# 论文图片资产索引

## 主文图片 (`main/`)

| 文件 | 编号 | 内容 | 位置 | 尺寸 |
|------|------|------|------|------|
| `fig0_cover_trajectories.png` | Cover | 3D 双螺旋汇聚（思维轨迹） | Graphical Abstract / Conclusion | 772 KB |
| `fig1_architecture.png` | Fig 1 | Mini-JMamba 架构图 | §2 Method | 134 KB |
| `fig2_transfer_matrix.png` | Fig 2 | 三域迁移矩阵 9×9 | §3.2 Transfer | 252 KB |
| `fig3_trajectory_comparison.png` | Fig 3 | IID vs OOD 崩溃轨迹 | §4.1 OOD Analysis | 429 KB |
| `fig4_endpoint_distribution.png` | Fig 4 | OOD 终点散点聚类 | §4.1 OOD Analysis | 53 KB |
| `fig5_tsae_resonance.png` | Fig 5 | TSAE 时间-频率共振热力图 | §4.2 Mechanism | 329 KB |
| `fig6_cross_domain.png` | Fig 6 | Audio/IPD/RF 三面板概念图 | §4.3 Cross-Domain | 408 KB |

**主文总计**: 7 张图，~2.4 MB

---

## 附录图片 (`supp/`)

### Appendix B: OOD Dynamics

| 文件 | 编号 | 内容 |
|------|------|------|
| `sup_fig_B1_temporal_norm.png` | Fig B1 | 时序范数演化 |
| `sup_fig_B2_layer_norms.png` | Fig B2 | 层间范数对比 |
| `sup_fig_B3_trajectory_detail.png` | Fig B3 | 6样本轨迹详细对比 |

### Appendix C: Multi-Task Analysis

| 文件 | 编号 | 内容 |
|------|------|------|
| `sup_fig_C1_tasks_overlay.png` | Fig C1 | IID/OOD_digits/OOD_length 叠加 |
| `sup_fig_C2_tasks_trajectories.png` | Fig C2 | 三任务分组轨迹 |

### Appendix D: Cross-Domain Details

| 文件 | 编号 | 内容 |
|------|------|------|
| `sup_fig_D1_cross_domain_2d.png` | Fig D1 | 2D 跨域轨迹精细对比 |

### Appendix E: Additional Analysis

| 文件 | 编号 | 内容 |
|------|------|------|
| `sup_fig_E1_attention.png` | Fig E1 | 注意力热图 |
| `sup_fig_E2_state_dynamics.png` | Fig E2 | 状态动力学分析 |
| `sup_fig_E3_pruning.png` | Fig E3 | 修剪效益分析 |

**附录静态图总计**: 9 张，~3.0 MB

---

## 补充视频 (`supp/`)

| 文件 | 编号 | 内容 | 大小 |
|------|------|------|------|
| `video_S1_ood_collapse.gif` | Video S1 | OOD 崩溃动态演化 | 5.3 MB |
| `video_S2_multi_task.gif` | Video S2 | 多任务轨迹动画 | 5.3 MB |
| `video_S3_cross_domain.gif` | Video S3 | 跨域同步演化 | 2.8 MB |
| `video_S4_thought_trajectories.gif` | Video S4 | 3D 思维轨迹动画 | 3.9 MB |

**视频总计**: 4 个 GIF，~17.3 MB

---

## 总计

| 类别 | 数量 | 大小 |
|------|------|------|
| 主文图片 | 7 张 | ~2.4 MB |
| 附录图片 | 9 张 | ~3.0 MB |
| 视频动画 | 4 个 | ~17.3 MB |
| **总计** | **20 个文件** | **~22.7 MB** |

---

## LaTeX 引用示例

### 主文

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.9\linewidth]{figures/main/fig3_trajectory_comparison.png}
  \vspace{-6pt}
  \caption{Hidden state trajectories for IID (green), OOD digits (orange), 
           and OOD length (red) samples. OOD length trajectories drift into 
           unexplored regions of the latent space.}
  \label{fig:trajectory}
\end{figure}
```

### 附录

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{figures/supp/sup_fig_B1_temporal_norm.png}
  \caption{Temporal evolution of hidden state L2 norms. OOD length samples 
           exhibit increased variance, indicating representational instability.}
  \label{fig:supp:temporal_norm}
\end{figure}
```

### 视频脚注

```latex
\footnotetext[‡]{Video S3 (\texttt{video\_S3\_cross\_domain.gif}) shows 
                 synchronized trajectory evolution across Audio and Optical domains.}
```

---

## 提交 Checklist

- [ ] 主文 `\includegraphics` 路径一致
- [ ] Figure 序号与文本交叉引用匹配
- [ ] Appendix 图从主文不再重复引用
- [ ] GIF 总大小 < 20 MB ✓ (17.3 MB)
- [ ] paper.bib 中无额外引用遗漏
- [ ] `latexmk` 通过 & 无 Overfull box

---

*Generated: 2026-01-02*

