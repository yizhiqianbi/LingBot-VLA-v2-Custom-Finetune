# LingBot-VLA-v2 Custom Fine-tuning

面向自采数据集 [`jokeru/take_wrong_item_right_arm`](https://huggingface.co/datasets/jokeru/take_wrong_item_right_arm) 的、可审计且可迁移的 [LingBot-VLA-v2](https://github.com/Robbyant/lingbot-vla-v2) 微调工程。

仓库只包含代码、配置模板、测试和文档，不包含数据、模型权重、归一化统计、训练 checkpoint 或 token。

## 当前状态

以下链路已在完整数据上实际执行：

- 固定 HF dataset revision 并下载完整 v2.1 数据。
- 审计 44 episodes、31,359 帧、44 parquet 和 176 个 MP4。
- 非破坏地转换为 LeRobot v3.0 训练副本。
- 使用官方 `FeatureTransform` 映射 7D 右臂、1D 夹爪和三路相机。
- 使用官方 `build_vla_dataset` 读取首、中、末三个 50-step action chunk。
- 计算全量 mean/std/quantile 归一化统计。
- 解码当前和未来三路图像，执行 Qwen3-VL 图像处理、语言 tokenization、55D padding 和 joint mask。

尚未执行 GPU optimizer step。当前验证节点没有可用 CUDA GPU，`train.sh` 会在启动 `torchrun` 前明确失败，不会把无 GPU 状态误报为训练成功。

## 数据映射

HF 元数据没有提供 15 维 state/action 的维度名称，因此映射分成“可直接验证”和“需要所有者确认”两层。

| 统一特征 | 原始切片 | 训练语义 | 处理 |
|---|---:|---|---|
| `observation.state.arm.position` | `observation.state[0:7]` | 右臂 7D 关节位置 | mean/std |
| `action.arm.position` | `action[0:7]` | 右臂 7D 目标 | 相对当前 state 的 joint delta |
| `observation.state.effector.position` | `observation.state[14:15]` | 右夹爪 1D 位置 | mean/std |
| `action.effector.position` | `action[14:15]` | 右夹爪绝对目标 | 不减 state |
| ignored | `[7:14]` | 推断为非任务臂/被动通道 | 不送入 policy |

全量数值审计证明：对每个非 terminal frame，`action[t, 7:15] == state[t+1, 7:15]`，最大绝对误差为 0；`action[0:7]` 与下一帧 state 独立，平均绝对误差约 0.00654。结合 task 文本和视频，当前映射是最一致的解释，但原始 metadata 的 `names` 为 `null`，所以正式训练前仍要求数据所有者确认。

相机映射：

| LingBot 相机 | 原始相机 |
|---|---|
| `camera_top` | `observation.images.right_eye` |
| `camera_wrist_left` | `observation.images.left_wrist` |
| `camera_wrist_right` | `observation.images.right_wrist` |

`left_eye` 为了适配公开模型的三相机接口而省略。完整依据见 [DATASET_AND_MAPPING.md](docs/DATASET_AND_MAPPING.md)。

## 流程

```text
HF v2.1 source
  -> source audit
  -> non-destructive v3.0 preparation + receipt
  -> owner layout acceptance
  -> rendered runtime configs
  -> normalization stats + manifest
  -> numeric loader smoke
  -> full image/language sample smoke
  -> 2-step GPU train smoke
  -> fine-tuning
```

每个阶段都有独立失败条件。训练只读取带有效 prepare receipt 的 v3.0 目录；正式训练还要求布局 acceptance 和与当前 contract 完全匹配的 norm manifest。

## 环境

推荐 Python 3.12，并使用官方仓库锁定提交：

```bash
scripts/bootstrap_upstream.sh
python -m pip install -r .upstream/lingbot-vla-v2/requirements.txt
python -m pip install --no-deps lerobot==0.4.2
python -m pip install -e .
```

这里对 `lerobot` 使用 `--no-deps`，因为 LingBot 官方依赖固定了 `datasets==3.6.0` 和 Torch 2.8，而 LeRobot 0.4.2 的发布依赖范围与其不完全一致。此组合已经通过 v2.1 转换、v3 loader、视频解码和完整 sample 测试。

模型文件按照官方训练要求准备在代码仓库之外：

- `robbyant/lingbot-vla-v2-6b`
- `Qwen/Qwen3-VL-4B-Instruct`
- `Ruicheng/moge-2-vitb-normal`
- LingBot depth checkpoint
- DINO-VIDEO teacher checkpoint 与 config

## 配置路径

以 `.env.example` 为模板设置环境变量：

```bash
LINGBOT_PYTHON=/path/to/python
LINGBOT_SOURCE_DATASET_ROOT=/data/take_wrong_item_right_arm
LINGBOT_TRAIN_DATASET_ROOT=/data/take_wrong_item_right_arm_v30
LINGBOT_MODEL_PATH=/models/lingbot-vla-v2-6b
LINGBOT_TOKENIZER_PATH=/models/Qwen3-VL-4B-Instruct
LINGBOT_MOGE_PATH=/models/moge/model.pt
LINGBOT_DEPTH_PATH=/models/lingbot-depth/model.pt
LINGBOT_VIDEO_CKPT_PATH=/models/dino-video/teacher_step_10000.pth
LINGBOT_VIDEO_CONFIG_PATH=/models/dino-video/config.yaml
LINGBOT_RUN_OUTPUT=/runs/take_wrong_item_right_arm
CUDA_VISIBLE_DEVICES=0,1,2,3
```

`LINGBOT_SOURCE_DATASET_ROOT` 是不可变的 HF v2.1 下载；`LINGBOT_TRAIN_DATASET_ROOT` 是转换后的 v3.0 训练副本。二者必须位于代码仓库之外且不能互相嵌套。

## 快速开始

1. 下载固定 revision：

```bash
scripts/download_dataset.sh
```

2. 审计全部 state/action 文件并抽样解码视频：

```bash
scripts/validate_dataset.sh --decode-videos
```

3. 生成非破坏的 v3.0 训练副本：

```bash
scripts/prepare_dataset.sh
```

4. 数据所有者核对 `configs/dataset_contract.yaml` 中的关节和相机语义后，写入 acceptance：

```bash
scripts/validate_dataset.sh \
  --decode-videos \
  --accept-inferred-layout
```

5. 生成 runtime config 和全量归一化统计：

```bash
scripts/render_configs.sh
scripts/compute_norm_stats.sh
```

6. 验证 numeric action chunk 和完整训练 sample：

```bash
scripts/smoke_loader.sh
scripts/smoke_full_sample.sh --index 0
```

7. GPU 环境与两步训练 smoke：

```bash
scripts/check_environment.sh --require-cuda
scripts/train_smoke.sh
```

8. 启动正式微调：

```bash
scripts/train.sh
```

训练参数可以附加在命令末尾，例如：

```bash
scripts/train.sh \
  --train.max_steps 8000 \
  --train.gradient_accumulation_steps 2 \
  --train.save_steps 1000
```

不要手工设置 `global_batch_size`。官方参数层会按照 `micro_batch_size × GPU 数 × gradient_accumulation_steps` 自动计算，避免更换 GPU 数量后配置失效。

## 技术 smoke 模式

在数据所有者确认前，可以验证数据工程，但不能启动正式训练：

```bash
export LINGBOT_ALLOW_UNCONFIRMED=1
scripts/render_configs.sh --allow-unconfirmed
scripts/compute_norm_stats.sh
scripts/smoke_loader.sh
scripts/smoke_full_sample.sh
```

这时 norm manifest 会记录 `layout_confirmed: false`，`train.sh` 会拒绝使用。完成正式 acceptance 后重新运行 `scripts/compute_norm_stats.sh` 即可生成可训练的统计文件。

## 代码结构

```text
configs/                  数据 contract、robot mapping 和训练模板
docs/                     数据、管线、训练和验证文档
scripts/                  可直接执行的端到端命令
src/lingbot_vla_finetune/ 审计、转换、统计、渲染和 smoke 实现
tests/                    不依赖真实数据的单元测试
upstream.lock             官方 LingBot-VLA-v2 固定提交
```

详细流程见 [PIPELINE.md](docs/PIPELINE.md)，训练参数和恢复策略见 [TRAINING.md](docs/TRAINING.md)，本次真实验证记录见 [VERIFICATION.md](docs/VERIFICATION.md)。

## 安全边界

- `.gitignore` 排除数据、视频、parquet、模型、checkpoint、token、runtime config 和统计文件。
- `download`、`prepare` 和 `export` 拒绝把大文件写进代码仓库。
- HF revision、LingBot upstream revision 和 LeRobot 版本均固定。
- `train.sh` 要求 layout acceptance、prepared receipt、norm manifest 和 CUDA。
- 通过 `bash -o pipefail` 执行官方训练脚本，保留 `torchrun` 的真实退出码。
