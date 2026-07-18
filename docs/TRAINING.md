# Fine-tuning Guide

## 1. 训练目标

本仓库实现的是 LingBot-VLA-v2 公开 6B checkpoint 的 downstream post-training，不修改模型结构。数据层将单臂任务映射到公开模型的统一 55D action/state head，并用 mask 屏蔽 47 个 padding 维度。

默认训练保留公开 checkpoint 所需的辅助监督：

- 当前 depth alignment。
- future depth alignment。
- DINO-VIDEO future/current patch alignment。
- 50-step flow-matching action objective。

因此 MoGe、LingBot depth 和 DINO-VIDEO teacher 权重都必须存在。

## 2. 默认策略

当前模板的关键参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `chunk_size` | 50 | 约 1.79 秒 action horizon |
| `max_state_dim` | 55 | 与公开模型一致 |
| `max_action_dim` | 55 | 与公开模型一致 |
| `loss_type` | `L1_fm` | 官方 RoboTwin 配置使用的 flow-matching loss |
| `optimizer` | AdamW | 小规模微调更保守 |
| `lr` | `1e-5` | 低于官方大规模训练配置 |
| `lr_decay_style` | cosine | 3% warmup |
| `micro_batch_size` | 1 | 降低显存压力 |
| `gradient_accumulation_steps` | 1 | 可按 GPU 数和目标 global batch 调整 |
| `max_steps` | 5000 | 初始上限，建议按验证表现调整 |
| `save_steps` | 500 | 周期 checkpoint |
| `freeze_vit` | true | 冻结 Qwen visual tower，降低过拟合和显存 |
| `use_future_image` | true | 为辅助 future supervision 提供未来帧 |

数据只有 44 episodes。默认配置刻意使用较低学习率并冻结 visual tower，但仍可能过拟合。建议保存多个 checkpoint，在独立 rollout 或至少 held-out episodes 上选择，而不是只看 training loss。

## 3. GPU 数与 batch size

不要在 YAML 中固定 `global_batch_size`。官方参数层自动计算：

```text
global_batch_size = micro_batch_size
                  * data_parallel_size
                  * gradient_accumulation_steps
```

例如 4 GPUs、micro batch 1、accumulation 2，对应 global batch 8：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train.sh \
  --train.gradient_accumulation_steps 2
```

切换到 1 或 8 GPUs 时不需要修改 YAML。

## 4. 建议的调试顺序

先执行：

```bash
scripts/check_environment.sh --require-cuda
scripts/smoke_loader.sh
scripts/smoke_full_sample.sh --index 0
```

再执行两步 optimizer smoke：

```bash
scripts/train_smoke.sh
```

两步 smoke 应检查：

- model 和所有 teacher 权重加载无 missing/unexpected key。
- rank 初始化完成。
- 首 batch 无视频 timestamp 错误。
- action/depth/video loss finite。
- backward、optimizer step 和 checkpoint 都成功。
- 每个 rank 正常退出，shell exit code 为 0。

只有这些条件满足后再增加 `max_steps`。

## 5. 推荐实验阶梯

### 5.1 Pipeline sanity

```bash
scripts/train_smoke.sh
```

2 steps，只验证工程链路。

### 5.2 Short overfit

```bash
scripts/train.sh \
  --train.max_steps 100 \
  --train.save_steps 100 \
  --train.use_wandb false
```

用于确认 loss 能下降、action 输出能拟合。不要把该 checkpoint 当最终模型。

### 5.3 Conservative fine-tune

```bash
scripts/train.sh \
  --train.max_steps 2000 \
  --train.save_steps 250 \
  --train.lr 1e-5
```

比较多个 checkpoint 的 held-out loss 和真实 rollout。

### 5.4 Extended run

只有在 2k steps 后仍无明显过拟合并且 rollout 改善时，再增加到 5k-8k steps。44 episodes 不支持仅凭训练 loss 无限制延长训练。

## 6. 数据划分建议

当前 HF 数据只有一个 task，官方 single-dataset loader 会使用全部 44 episodes。严谨评估建议在数据发布侧建立 train/validation revision，按 episode 划分而不是按 frame 随机划分，避免相邻帧泄漏。

建议至少保留 4-8 个完整 episodes 作为 validation。若不修改 HF 数据，可以复制 v3 prepared 数据并构建 episode subset，但必须为 subset 建立独立 contract、receipt 和 norm stats，不能复用全量统计 manifest。

## 7. Checkpoint 和恢复

默认输出位于 `LINGBOT_RUN_OUTPUT`，必须在代码仓库之外。训练配置启用：

```yaml
enable_resume: true
ckpt_manager: dcp
save_hf_weights: true
```

重复启动相同 output dir 时，官方 checkpointer 会尝试恢复。开始全新实验时使用新的 output dir，避免意外续训旧 optimizer state。

每次实验应保留：

- rendered runtime config。
- runtime manifest。
- norm stats manifest。
- Git commit SHA。
- upstream commit SHA。
- GPU 型号与数量。
- 实际命令行 overrides。

这些信息足以定位数据、模型和训练参数差异。

## 8. 部署动作反变换

模型输出先按 mask 取出 8 个 active dims，再反归一化：

```text
right_arm_target = current_right_arm_state + predicted_arm_delta
right_gripper_target = predicted_absolute_gripper
```

部署侧必须使用与训练相同的关节顺序和单位。若机器人控制器使用角度、归一化 encoder count 或不同关节顺序，必须在执行前做显式变换，不能直接发送模型输出。

## 9. 常见失败

### `BackwardCompatibilityError` for v2.1

训练误用了 source 目录。设置 `LINGBOT_TRAIN_DATASET_ROOT` 指向 `scripts/prepare_dataset.sh` 生成的 v3.0 目录。

### `Stale layout acceptance`

Contract 已修改。重新审计并由所有者确认后生成 acceptance。

### `Normalization stats were computed before layout confirmation`

技术 smoke 使用了 unconfirmed 模式。完成 acceptance 后重新执行 `scripts/compute_norm_stats.sh`。

### CUDA unavailable

数据处理可以在 CPU 执行，LingBot-VLA-v2 训练和 fused MoE kernel 需要 CUDA GPU。

### `global_batch_size` mismatch

删除手工 override，仅设置 `gradient_accumulation_steps`。

### Video timestamp tolerance error

先核对 prepared receipt 和 `meta/episodes` 中的 from/to timestamps。不要直接增大 tolerance 掩盖错误；应确认转换后的合并视频顺序和 episode metadata 一致。
