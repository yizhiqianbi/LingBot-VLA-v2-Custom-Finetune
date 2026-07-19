# Preprocessing and Training Pipeline

[文档索引](../README.md)

本文只维护从固定依赖、下载、source audit、v3 preparation 到训练启动前检查的执行流程。数据语义见 [Dataset and Mapping](../data/DATASET_AND_MAPPING.md)，训练参数见 [Fine-tuning](../training/FINETUNING.md)，当前通过/待验状态见 [Validation Status](../reference/VALIDATION_STATUS.md)。

## 1. 设计目标

这条管线解决四类风险：

- HF 数据 revision 漂移。
- LeRobot v2.1 与当前 LingBot/LeRobot loader 不兼容。
- 15D 原始 action 没有维度名称，容易误训被动通道。
- runtime config、norm stats 和训练 checkpoint 之间缺少可追溯关系。

所有大文件都位于代码仓库之外，仓库可以直接打包迁移到其他服务器。

## 2. Stage 0：固定依赖

`upstream.lock` 固定 LingBot-VLA-v2 Git commit。`scripts/bootstrap_upstream.sh` 将官方仓库 clone 到被 `.gitignore` 排除的 `.upstream/lingbot-vla-v2`，并使用 detached HEAD。

Contract 固定：

- HF repo id 和 commit SHA。
- LeRobot source/prepared version。
- 原始 feature schema。
- 动作和相机映射。
- 数据审计阈值。

`render` 会检查官方 checkout 的实际 HEAD，版本不一致时立即失败。

## 3. Stage 1：下载

入口：

```bash
scripts/download_dataset.sh
```

行为：

- 使用 `snapshot_download(repo_type="dataset")`。
- 强制使用 contract 中的 40 字符 revision。
- 支持可选 `HF_TOKEN_FILE`，但不读取或打印 token 内容。
- 目标目录必须在代码仓库之外。
- Hugging Face 自带 metadata receipt 用于后续 revision 校验。

重复执行是幂等的，Hugging Face 会续传或复用完整文件。

## 4. Stage 2：source audit

入口：

```bash
scripts/validate_dataset.sh --decode-videos
```

硬检查：

- `info.json`、`episodes.jsonl`、`tasks.jsonl` 存在。
- HF local revision 与 contract 一致。
- format、FPS、episode/frame 数一致。
- state/action width 均为 15。
- 4 个 camera feature 齐全。
- episode index 集合严格为 `0..43`，无重复。
- 每个 parquet 行数、frame index、episode index、task index 正确。
- state/action 全部 finite。
- 时间戳单调，FPS 偏差超过 5% 才 hard fail。
- `action[7:15] == next_state[7:15]` 关系满足阈值。
- `action[0:7]` 不是 next-state copy。
- 176 个视频文件存在且非空。
- 开启 decode 时抽样验证 OpenCV 可打开、可解码且 frame count 合理。

输出：

```text
work/audit_report.json
```

该目录被 git 忽略。

## 5. Stage 3：v2.1 -> v3.0 preparation

入口：

```bash
scripts/prepare_dataset.sh
```

为什么转换：当前官方 LingBot README 声称支持 v2.1 和 v3.0，但配合 LeRobot 0.4.2 加载本数据时，v2.1 会触发 `BackwardCompatibilityError`。管线使用 LeRobot 官方 v3 converter 的组成函数，在独立目录生成训练副本。

转换是非破坏性的：

- source v2.1 只读。
- 输出先写入同级唯一 staging 目录。
- data parquet、视频和元数据全部完成并校验后才原子 rename。
- 中途失败会清理 staging，不留下被误认为完整的数据目录。

后置检查：

- `codebase_version == v3.0`。
- episode/frame/FPS 与 contract 一致。
- state/action/camera features 齐全。
- state/action width 正确。
- data、video、episode metadata、tasks 和 stats 文件存在。

输出目录包含：

```text
.lingbot_vla_prepare_receipt.json
```

Receipt 记录 source revision、contract hash、LeRobot 版本、转换参数、文件计数和 source audit 摘要。有效输出重复执行时直接复用；只有 contract hash 变化且其他身份/结构完全一致时，才重新审计并刷新 receipt。

## 6. Stage 4：layout acceptance

入口：

```bash
scripts/validate_dataset.sh --decode-videos --accept-inferred-layout
```

只有 source audit 通过时才能写 acceptance。Acceptance 包含：

- contract SHA256。
- dataset repo/revision。
- source dataset absolute path。
- 明确列出的语义假设。
- acceptance 时间。

任何 mapping 或 contract 修改都会使 acceptance 失效。`render` 还会检查 acceptance 的 source path 与 prepare receipt 一致。

## 7. Stage 5：runtime config render

入口：

```bash
scripts/render_configs.sh
```

模板位于 `configs/`，绝对路径只写到被忽略的 `work/runtime/`：

```text
work/runtime/take_wrong_item_right_arm.yaml
work/runtime/configs/robot_configs/take_wrong_item_right_arm.yaml
work/runtime/runtime_manifest.json
```

Render 检查：

- layout acceptance。
- prepare receipt。
- upstream commit。
- 数据、主模型、tokenizer、MoGe、depth 和 video teacher 路径。
- 若用于训练，还检查 norm stats 与 manifest。

## 8. Stage 6：normalization

入口：

```bash
scripts/compute_norm_stats.sh
```

官方 `scripts/compute_norm_stats.py` 会导入完整训练 dataclass，继而导入 MoE Triton kernel，纯数据统计也要求 CUDA。本仓库提供 data-only 等价实现：

- 仍调用官方 `build_vla_dataset`。
- 仍调用官方 `FeatureTransform`。
- 仍调用官方 `RunningStats` 和 `normalize.save`。
- 不导入模型、MoE kernel 或 CUDA。
- 使用 DataLoader workers 遍历完整映射后数据。

输出：

```text
work/norm_stats/take_wrong_item_right_arm.json
work/norm_stats/take_wrong_item_right_arm.json.manifest.json
```

Manifest 中的 SHA256、contract、dataset、upstream、chunk size 或 confirmation 不匹配时，正式训练会失败。

## 9. Stage 7：loader smoke

Numeric smoke：

```bash
scripts/smoke_loader.sh
```

读取 dataset 首、中、末三个位置，检查：

- 7D arm state。
- 1D gripper state。
- `50 x 7` arm actions。
- `50 x 1` gripper actions。
- 全部 finite。

Full sample smoke：

```bash
scripts/smoke_full_sample.sh --index 0
```

额外执行：

- v3 合并视频按 episode timestamp 定位。
- 三路 current/future frame 解码。
- Qwen3-VL image processor。
- task tokenization。
- mean/std normalization。
- 55D state/action padding 与 mask。
- future depth/video teacher 所需的 uint8 image tensors。

## 10. Stage 8：training

两步 smoke：

```bash
scripts/train_smoke.sh
```

正式训练：

```bash
scripts/train.sh
```

训练前强制检查：

- CUDA 可用，且 `CUDA_VISIBLE_DEVICES` 已设置。
- LeRobot 版本为 0.4.2。
- upstream revision 正确。
- runtime config、acceptance、prepare receipt、norm manifest 全部有效。

官方 `train.sh` 内部使用 `torchrun | tee`。本仓库通过 `bash -o pipefail` 调用，避免 `torchrun` 失败后由 `tee` 返回 0。

## 11. 迁移与导出

代码包：

```bash
scripts/export_code.sh /tmp/LingBot-VLA-v2-Custom-Finetune.tar.gz
```

导出脚本排除 Git metadata、upstream clone、work、数据、视频、parquet、模型、checkpoint 和 token。迁移到新服务器后重新设置环境变量，并按 stage 顺序执行即可。
