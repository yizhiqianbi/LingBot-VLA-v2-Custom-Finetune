# LingBot-VLA-v2 Custom Fine-tuning

面向自采数据集 [`jokeru/take_wrong_item_right_arm`](https://huggingface.co/datasets/jokeru/take_wrong_item_right_arm) 的 LingBot-VLA-v2 微调、评测和部署前验证工程。

仓库只保存代码、配置模板、测试和文档，不提交数据、模型权重、归一化统计、checkpoint、运行产物或 token。官方 LingBot-VLA-v2 源码按 `upstream.lock` 固定版本，并下载到被 Git 忽略的 `.upstream/`。

> 当前能力边界：44 个 episodes 已完成数据审计、LeRobot v3.0 转换、完整 sample 验证、8-GPU smoke 和 2000-step 微调。step 2000 的训练集 replay MSE/MAE 为 `0.007354/0.051602`；独立 held-out、shadow test 和真机闭环仍未完成。最新证据以 [Validation Status](docs/reference/VALIDATION_STATUS.md) 为准。

## 数据合同

原始 state/action 均为 15D，但 metadata 没有维度名称。当前经数值和视频审计采用以下训练映射：

| 训练特征 | 原始切片 | 模型语义 |
|---|---:|---|
| arm state | `observation.state[0:7]` | 右臂 7D 关节位置 |
| arm action | `action[0:7]` | 相对当前 state 的 joint delta |
| gripper state | `observation.state[14:15]` | 右夹爪 1D 位置 |
| gripper action | `action[14:15]` | 右夹爪绝对目标 |
| ignored | `[7:14]` | 不送入当前任务 policy |

映射后的 8 个 active dimensions 会 pad 到 LingBot-VLA-v2 的 55D head，并通过 joint mask 屏蔽其余维度。

采集时四路相机线缆接反，raw key 只能作为稳定 stream ID，不能按名字推断物理位置。当前训练使用：

| LingBot 输入 | Raw stream |
|---|---|
| `camera_top` | `observation.images.left_eye` |
| `camera_wrist_left` | `observation.images.right_eye` |
| `camera_wrist_right` | `observation.images.right_wrist` |

部署必须保持同一映射；物理左右和修线后的迁移规则见 [Dataset and Mapping](docs/data/DATASET_AND_MAPPING.md)。

## 快速开始

```bash
git clone git@github.com:yizhiqianbi/LingBot-VLA-v2-Custom-Finetune.git
cd LingBot-VLA-v2-Custom-Finetune

scripts/bootstrap_upstream.sh
python -m pip install -r .upstream/lingbot-vla-v2/requirements.txt
python -m pip install --no-deps lerobot==0.4.2
python -m pip install -e .
```

以 `.env.example` 为模板配置 Python、数据、基础模型、teacher 权重和外部输出目录，然后执行：

```bash
scripts/download_dataset.sh
scripts/validate_dataset.sh --decode-videos
scripts/prepare_dataset.sh

# 数据所有者确认 action 和相机合同后执行
scripts/validate_dataset.sh --decode-videos --accept-inferred-layout

scripts/render_configs.sh
scripts/compute_norm_stats.sh
scripts/smoke_loader.sh
scripts/smoke_full_sample.sh --index 0
scripts/check_environment.sh --require-cuda
scripts/train_smoke.sh
scripts/train.sh
```

训练结束后使用 HF checkpoint 做 open-loop replay：

```bash
export LINGBOT_EVAL_STEP=2000
export LINGBOT_EVAL_TRAJ_IDS="0 10 20 30 43"
export CUDA_VISIBLE_DEVICES=0
scripts/eval_open_loop.sh
```

当前 replay trajectories 参与过训练，只能验证模型加载、预处理、反归一化和拟合链路，不能作为泛化或真机成功率。

## 文档入口

完整导航见 [Documentation Index](docs/README.md)。建议按以下顺序阅读：

1. [Validation Status](docs/reference/VALIDATION_STATUS.md)：已验证结果与剩余验证。
2. [Roadmap](docs/ROADMAP.md)：held-out、checkpoint 选择和真机测试顺序。
3. [Dataset and Mapping](docs/data/DATASET_AND_MAPPING.md)：15D action/state 与错线相机合同。
4. [Pipeline](docs/workflow/PIPELINE.md)：下载、审计、v3 转换、统计和训练前门槛。
5. [Fine-tuning](docs/training/FINETUNING.md)：训练策略、参数、恢复和常见失败。
6. [Evaluation and Deployment](docs/evaluation/EVALUATION_AND_DEPLOYMENT.md)：replay、shadow test 和 closed-loop。

## 仓库结构

```text
configs/                  数据合同、机器人映射和训练模板
docs/                     文档索引、路线图和专题文档
scripts/                  端到端命令入口
src/lingbot_vla_finetune/ 审计、转换、统计、渲染和 smoke 实现
tests/                    不依赖真实数据的单元测试
upstream.lock             官方 LingBot-VLA-v2 固定提交
work/                     本地生成物，不提交 Git
```

## 验证与导出

```bash
python -m unittest discover -s tests -v
scripts/export_code.sh /tmp/LingBot-VLA-v2-Custom-Finetune.tar.gz
```

`train.sh` 会检查 CUDA、上游 revision、layout acceptance、prepare receipt、runtime config 和 norm manifest。任何 mapping 变化都必须重新审计、确认并计算 train-only normalization stats。
