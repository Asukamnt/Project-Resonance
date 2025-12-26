## Change log（本次修订摘要）
- 调整 S1 为最终发布（Final Gate）验收项并增加 Stage A 与 Final 的分层说明。
- 保持 S2 为 Stage A(MVP) 核心验收入口，补充 Stage A vs Final 关系说明。
- 重写 S3 以区分逻辑数据规范与实现形态，兼容 manifest/on-the-fly 两种模式。
- 更新 S24/S25/S26 的验收方式为人工可执行检查，相关自检脚本改列为非必做建议。
- 更新阶段A检查表注记，使其与 S1/S3/S24-S26 的分层要求保持一致。

## Overview / MVP / Timeline

**MUST**
- S1 [p1] 【Final Gate】“端到端系统必须在纯音频域同时解决镜像复制、括号合法性与模运算三项任务，不得出现文本中间表征” -> 验收：运行 `python evaluate.py --stage final --tasks mirror bracket mod --no_text --report reports/system_overview_final.json`；产物：`reports/system_overview_final.json`
  - Stage说明：Stage A 不以此项为门槛，该条仅在 Final Release（Stage Final）时执行。
- S2 [p1] “MVP需在单块GPU上打通Task1镜像任务的数据生成器、Mini-JMamba模型与音频评分器，实现端到端闭环” -> 验收：运行 `python train.py --task mirror --stage mvp --device cuda:0 --logdir runs/mvp_task1`；产物：`runs/mvp_task1/`
  - Stage说明：满足本项即完成 Stage A（Week1）验收，后续阶段在此基础上递进直至满足 S1。

**SHOULD**
- [p1] MVP应在IID镜像任务上达到>95%准确率，并展示对未见符号和更长序列的复制能力。
- [p1-p2] 时间线建议按周推进：周1数据与评分器、周2模型集成、周3-7逐步引入任务2/3与加难度、周8集成与文档/发布。

**OPTIONAL**
- 暂无PDF提出的可选事项。

## Data (Task1/Task2/Task3, IID/OOD, negative controls, splits)

**MUST**
- S3 [p2] “所有任务数据在逻辑上需等价于 (input_wave, output_wave, transcripts, difficulty, split, seed/manifest) 的成对样本，并能通过 manifest+seed 复现” -> 验收：运行 `python data/generate_all.py --manifest data/manifest.json --seed 42 --mode manifest_only`；产物：`data/manifest.json`
  - 模式A（批量落盘）：可追加 `python data/generate_all.py --manifest data/manifest.json --mode export_wave --out-dir data/wave/` 生成16kHz波形文件。
  - 模式B（按需合成）：保存符号序列与合成脚本，执行 `python data/synthesize_on_the_fly.py --manifest data/manifest.json --seed 42` 时需无差异复现波形。
- S4 [p3-p4] “Task1训练需使用至少10个基础符号、为每个符号分配唯一频率并在实例中随机相位/微扰，同时准备OOD长度与OOD新符号测试集” -> 验收：运行 `python data/generate_task1.py --alphabet train:A-J --ood_symbols K,L --max_train_len 10 --ood_lens 15 20 --manifest data/task1/manifest.json`；产物：`data/task1/`
- S5 [p4-p5] “Task2数据必须用两种括号音调编码、保证有效/无效样本各半并覆盖更长深度的OOD长度评估集” -> 验收：运行 `python data/generate_task2.py --balance 0.5 --train_max_len 10 --ood_lens 12 14 --manifest data/task2/manifest.json`；产物：`data/task2/`
- S6 [p5-p7] “Task3数据需覆盖单步与多步模运算，使用0-9与‘%’音调编码，提供更大数字及额外组成深度的OOD集合” -> 验收：运行 `python data/generate_task3.py --steps 1 2 --digits 0-9 --ood_compose 3 --ood_digits 5 --manifest data/task3/manifest.json`；产物：`data/task3/`
- S7 [p6-p7] “需生成负向对照数据（随机映射、标签置换、相位扰动等）以验证模型未走捷径” -> 验收：运行 `python data/generate_negative_controls.py --include phase_shuffle label_shuffle --manifest data/negative_controls/manifest.json`；产物：`data/negative_controls/`
- S8 [p7-p8] “必须建立独立的Train/Val/Test IID划分与OOD长度/符号/组合测试集，并固定随机种子以可复现” -> 验收：运行 `python evaluate.py --summarize_splits --seeds 42 --manifest data/manifest.json --out reports/data_splits.md`；产物：`reports/data_splits.md`

**SHOULD**
- [p3] Task1后期可引入重复两遍输出或反序镜像作为更高难度课程。
- [p5] Task2可视需求加入二进制逻辑门子任务以辅助学习。
- [p6] 若选择按需生成，可缓存符号序列或谱特征以兼顾存储与性能。

**OPTIONAL**
- [p6] 可选地预计算并缓存梅尔谱或其他特征以加速训练。

## Model (Mini-JMamba结构与输入输出表征)

**MUST**
- S9 [p8-p9] “Mini-JMamba模型必须由10层Mamba-2 SSM和2层注意力组成，总计12层” -> 验收：运行 `python scripts/inspect_model.py --expect_ssm 10 --expect_attn 2`；产物：`model/mini_jmamba.py`
- S10 [p9-p10] “输入端需包含将16kHz音频降采到~100Hz帧的频谱或卷积编码器（例如10ms步长的80维梅尔谱）并做归一化” -> 验收：运行 `python tests/test_frontend.py --check-mel 80 --frame_hop 0.01`；产物：`model/frontend.py`
- S11 [p10-p12] “解码端需输出与目标一致的时间帧级谱表示并支持可选ISTFT回波形，用于主损失计算” -> 验收：运行 `python tests/test_decoder.py --expect-head spectrogram --istft`；产物：`model/decoder.py`
- S12 [p12-p13] “模型需提供分支以输出符号级表示供CTC辅助头读取，并统一处理多任务输出” -> 验收：运行 `python tests/test_aux_head.py --head ctc --vocab unified`；产物：`model/heads.py`

**SHOULD**
- [p9-p10] 编码端的滑窗自注意力应限制在局部窗口以保持线性复杂度。
- [p10-p12] 建议在解码阶段使用下采样后的跨注意力以检索编码记忆。

**OPTIONAL**
- [p9] 可探索利用SSM状态续接编码与解码的单通道运行模式。
- [p10] 若需要更高音质，可尝试直接生成波形并追加轻量声码器。

## Losses

**MUST**
- S13 [p14-p15] “训练必须联合时间域L1损失与多分辨率STFT损失以约束输出音频频谱” -> 验收：运行 `python tests/test_losses.py --expect l1 stft_multi`；产物：`model/losses.py`
- S14 [p15-p16] “需加入符号级CTC辅助损失（覆盖Task1/3）与Task2的二分类交叉熵，并与主损失权重化组合” -> 验收：运行 `python tests/test_losses.py --expect ctc binary_ce --weights alpha=1 beta=1 gamma=0.5 delta=0.5`；产物：`model/losses.py`

**SHOULD**
- [p16] 建议在训练早期提高CTC权重、后期提升音频重建权重以加速收敛。
- [p16] 可监控高频能量并按需加入惩罚以防播放不可闻信息。

**OPTIONAL**
- [p15] 若具备对齐信息，可选用分段交叉熵替代CTC。

## Training plan

**MUST**
- S15 [p17-p19] “训练需采用动态批次与逐步课程设计，按任务长度/难度递增并维持任务间平衡” -> 验收：运行 `python train.py --config configs/curriculum.yaml --dry-run`；产物：`configs/curriculum.yaml`
- S16 [p17-p18] “优化必须使用AMP混合精度、AdamW优化器、梯度裁剪以及带预热的余弦学习率调度” -> 验收：运行 `python tests/test_trainer.py --check-amp --optimizer adamw --lr_schedule cosine_warmup --grad_clip 1.0`；产物：`trainer/optimization.py`
- S17 [p18-p20] “需实现多任务Subject-Selector采样器，按可配置概率在Task1/2/3之间轮换并可动态调节” -> 验收：运行 `python tests/test_selector.py --tasks mirror bracket mod --adaptive`；产物：`dataloaders/subject_selector.py`
- S18 [p19-p21] “训练过程中必须定期在各任务的IID与OOD验证集上记录EM/Token等指标并生成日志与检查点” -> 验收：运行 `python evaluate.py --from-checkpoint runs/latest.ckpt --splits val_iid,ood --metrics em,token --log reports/val_history.json`；产物：`reports/val_history.json`

**SHOULD**
- [p18] 建议使用梯度累积以在长序列时保持有效批量。
- [p19-p20] 建议根据任务表现自动调整采样概率避免遗忘。
- [p20] 推荐使用W&B或TensorBoard跟踪损失与指标。

**OPTIONAL**
- [p19] 如多任务难以收敛，可选用任务分阶段微调或蒸馏策略。

## Evaluation & diagnosis

**MUST**
- S19 [p21-p22] “评估必须报告Exact Match、Token Accuracy与编辑距离等指标” -> 验收：运行 `python evaluate.py --splits test_iid --metrics em token edit --output reports/test_metrics.json`；产物：`reports/test_metrics.json`
- S20 [p21-p22] “需在IID、OOD长度、OOD符号与OOD组合等全部拆分上评估并比较性能” -> 验收：运行 `python evaluate.py --splits test_iid,ood_length,ood_symbol,ood_compose --summary reports/ood_summary.md`；产物：`reports/ood_summary.md`
- S21 [p22-p23] “必须区分推理与渲染错误，利用评分器与CTC头记录替换/插删/时序等分类并输出诊断报告” -> 验收：运行 `python diagnostics/analyze_errors.py --use-ctc --report reports/error_analysis.md`；产物：`reports/error_analysis.md`
- S22 [p23-p24] “需完成至少五项消融实验（无注意力、无辅助损失、输入表示变体、下采样因子变化、无课程等）并整理对比结果” -> 验收：运行 `python experiments/run_ablations.py --suite core5 --report reports/ablations.csv`；产物：`reports/ablations.csv`

**SHOULD**
- [p22-p23] 建议可视化预测谱图与目标谱图以定位渲染问题。
- [p23] 推荐在错误集上检查隐藏状态与符号计数的相关性。

**OPTIONAL**
- [p24] 可针对特定失败模式追加定制数据集进行再训练测试。

## Risks

**MUST**
- S23 [p24-p28] “必须维护涵盖至少十大风险（过拟合/捷径、辅助作弊、资源/OOM、训练不稳、容量不足、多任务干扰、评分器偏差、进度延误、SSM实现问题、音频质量）的风险日志与缓解执行情况” -> 验收：运行 `python docs/update_risk_log.py --expected 10`；产物：`docs/risk_log.md`

**SHOULD**
- [p25-p27] 建议按风险类别落实相应缓解措施（负向对照、相位扰动、梯度裁剪、任务重平衡等）并在日志中记录状态。

**OPTIONAL**
- [p28] 可选引入判别器或后处理滤波以提升音质。

## Open-source plan (repo structure, milestones)

**MUST**
- S24 [p29-p30] “必须按预定结构创建开源仓库（含README、requirements、data生成脚本、model模块、train/evaluate脚本与实验配置目录）” -> 验收：运行 `python -c "import pathlib,sys;root=pathlib.Path('.');required=['README.md','requirements.txt','data','model','train.py','evaluate.py','experiments'];missing=[p for p in required if not (root/p).exists()];sys.exit(len(missing)!=0)"`；产物：`README.md`
  - 建议脚本（非必需）：可保留 `scripts/check_repo_structure.py` 以自动化校验，但不作为验收前置条件。
- S25 [p29-p31] “需按时间线公开阶段成果：周1占坑代码、周3 Task1 MVP演示、周5多任务版本、周8最终发布” -> 验收：运行 `python -c "from pathlib import Path;log=Path('docs/milestone_log.md');assert log.exists();print(log.read_text())"` 并确认里程碑条目含状态列；产物：`docs/milestone_log.md`
  - Stage说明：允许里程碑记录当前达成/未达成状态，Stage A 可仅勾选 Week1/Week3 计划，最终发布需补齐全部节点。
- S26 [p30] “必须提供复现资产（固定随机种子、配置文件、检查点、示例音频）以支持外部复现” -> 验收：运行 `python -c "import pathlib,sys;paths=['configs','artifacts/checkpoints','artifacts/audio_examples','docs/repro_seeds.json'];missing=[p for p in paths if not pathlib.Path(p).exists()];sys.exit(len(missing)!=0)"`；产物：`artifacts/`
  - 建议脚本（非必需）：如需自动验证，可使用 `scripts/verify_repro_assets.py`，但验收以人工可执行检查为准。

**SHOULD**
- [p30] README应包含模型描述、实验结果表与示例音频链接。
- [p31] 建议准备技术报告或博客同步发布关键样例。
- [p29-p31] 可保留自动化检测脚本（如 check_repo_structure / check_milestones）以提升团队协作效率。

**OPTIONAL**
- [p31] 可考虑后续投稿工作坊或发布更长久的演示站点。

### 阶段A完成检查表
- [x] Overview / MVP / Timeline（Stage A 以 S2 为验收，S1 属 Final Gate）
- [x] Data（S3 支持 manifest/on-the-fly 双模式并已对齐 Stage A 需求）
- [x] Model
- [x] Losses
- [x] Training plan
- [x] Evaluation & diagnosis
- [x] Risks
- [x] Open-source plan（S24-S26 调整为人工可执行验收且兼容 Stage A 进度）

