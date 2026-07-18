# Evaluation and Deployment Checks

本文说明如何验证微调 checkpoint 是否可加载、是否能在记录数据上输出合理动作，以及进入真机闭环前必须完成的检查。

## 1. 评测边界

当前 `jokeru/take_wrong_item_right_arm` 只有 44 个 episodes，正式 2000-step run 使用了全部数据。因此本文记录的 open-loop 指标是 **training replay**：它能验证模型加载、图像/语言预处理、动作反归一化和基本拟合能力，但不能估计未见场景上的泛化或真机成功率。

正式模型选择仍需要二者之一：

- 按 episode 留出 4-8 条 validation trajectories，并仅用 training subset 计算 norm stats。
- 在固定初始条件和安全约束下进行真实机器人 rollout，报告成功率和干预率。

禁止按 frame 随机拆分。相邻视频帧高度相关，会造成严重数据泄漏。

## 2. Checkpoint 结构

训练同时保存 DCP 恢复状态和 Hugging Face 推理权重：

```text
$LINGBOT_RUN_OUTPUT/checkpoints/global_step_2000/
  model/                         # DCP model state，用于恢复训练
  optimizer/                     # DCP optimizer state
  extra_state/
  hf_ckpt/
    model.safetensors.index.json
    model-00001-of-00006.safetensors
    ...
```

推理必须把 `hf_ckpt/` 传给 `--model_path`，不能传 `global_step_2000/` 或 `model/`。

## 3. Open-loop 命令

准备与训练一致的环境变量，然后运行：

```bash
export CUDA_VISIBLE_DEVICES=0
export LINGBOT_RUN_OUTPUT=/path/to/run
export LINGBOT_EVAL_STEP=2000
export LINGBOT_EVAL_TRAJ_IDS="0 10 20 30 43"
export LINGBOT_EVAL_MAX_INFER_TIME=3

scripts/eval_open_loop.sh
```

可用 `LINGBOT_EVAL_MODEL_PATH` 直接指定任意兼容的 `hf_ckpt`。脚本会验证模型索引、safetensors shards、训练配置、robot config、norm stats 和数据目录，随后从 rendered runtime 目录调用官方 `open_loop_eval.py`。

输出默认位于：

```text
$LINGBOT_RUN_OUTPUT/eval/open_loop_global_step_2000/
  eval.log
  0.png
  10.png
  ...
```

每张图逐维比较 unnormalized ground-truth action 和 prediction。日志报告每个 episode 及全部 episode 的 MSE/MAE。

## 4. 已完成的真实评测

2026-07-18 在一张 NVIDIA H200 上，以 BF16、eager attention、10 denoising steps 对 episodes `0, 10, 20, 30, 43` 进行相同协议测试。每条 episode 预测 3 个 50-step chunks，总计 750 个动作位置。

| Checkpoint | Average MSE | Average MAE |
|---|---:|---:|
| step 1500 | 0.008946 | 0.057300 |
| step 2000 | **0.007354** | **0.051602** |

step 2000 相对 step 1500 的 MSE 下降约 17.8%，MAE 下降约 9.9%。在该 replay 协议下，step 2000 是更好的部署候选。

step 2000 的逐 episode 结果：

| Episode | MSE | MAE |
|---:|---:|---:|
| 0 | 0.009117 | 0.052729 |
| 10 | 0.003245 | 0.037989 |
| 20 | 0.008819 | 0.056794 |
| 30 | 0.007911 | 0.055189 |
| 43 | 0.007676 | 0.055308 |

首个 forward 包含 warm-up，约 1.57 秒；后续单 batch、50-step chunk 稳定在约 0.66 秒。输出曲线能跟随多数右臂关节的整体趋势，但部分维度在 50-step 重规划边界存在跳变，夹爪维度存在小幅抖动。真机阶段不应直接执行完整 50-step chunk，应采用更短执行窗口并周期性重规划。

## 5. Shadow test

真机首次测试只读取传感器并记录预测，不下发控制命令。模型输入必须沿用训练时按画面内容确定的 stream mapping：

```text
model raw key observation.images.left_eye   <- head/global A
model raw key observation.images.right_eye  <- wrist A
model raw key observation.images.right_wrist <- wrist B
```

线缆修正或驱动重命名后，也必须将新的物理相机按上述视觉角色路由到模型 key。不能根据新名字直接替换。

输入还包括 15D `observation.state` 和训练时的精确 task 文本。模型反变换后返回紧凑 8D `action`：

```text
action[0:7] = right-arm absolute joint target
action[7]   = right-gripper absolute target
```

原始 `[7:14]` 未训练通道必须保持当前值，不能由模型输出覆盖。

Shadow test 至少检查：

- 所有动作 finite。
- 关节顺序、单位和正负方向与控制器一致。
- 目标位置在硬件 joint limits 内。
- 单步位置差、速度、加速度和 jerk 不超过训练数据及硬件阈值。
- 三路图像的内容角色与训练一致。
- 夹爪开合方向和数值范围正确。

## 6. Closed-loop rollout

通过 shadow test 后，从低速度、缩小 workspace、急停可用的条件开始。数据 nominal FPS 为 28；模型生成 50-step horizon，但首次闭环建议只执行前 1-4 步后重新规划，并对命令做限幅和低通处理。

每个 checkpoint 使用完全相同的初始条件，至少记录：

- 完整任务成功率。
- 正确抓取率、正确分类放置率。
- 掉落、碰撞、错误抓取次数。
- 人工干预/急停率。
- 完成时间和重规划次数。

最终 checkpoint 由 validation 或真实 rollout 选择，不能仅依据 training loss 或 replay MAE。
