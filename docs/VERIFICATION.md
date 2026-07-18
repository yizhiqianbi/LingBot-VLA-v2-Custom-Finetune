# Verification Record

验证日期：2026-07-18 UTC。

## 固定输入

- LingBot-VLA-v2 upstream: `2838c1862bbec1ea47942fb61512130f635eb595`
- Dataset: `jokeru/take_wrong_item_right_arm`
- Dataset revision: `76a008f11dc65a99d176061fb59807a9abdfdffc`
- LeRobot: 0.4.2
- Source format: v2.1
- Prepared format: v3.0

## Source audit

| 指标 | 结果 |
|---|---:|
| episodes | 44 |
| frames/parquet rows | 31,359 |
| parquet files | 44 |
| source MP4 files | 176 |
| source video bytes | 1,817,317,825 |
| cameras | 4 |
| state/action width | 15 / 15 |
| passive next-state max error | 0.0 |
| active action-next-state mean error | 0.0065421742 |
| active action-next-state max error | 0.4777231216 |
| timestamp FPS median | 28.3836 |
| hard errors | 0 |

Warning：episode 11 的 timestamp FPS 为 28.6152，相对 nominal 28 偏差 2.20%。

## v3 preparation

转换后：

- 1 个合并 data parquet。
- 1 个 episodes metadata parquet。
- 1 个 tasks parquet。
- 1 个 global stats JSON。
- 5 个合并 MP4；`right_eye` 因超过 500 MB 阈值拆成两个文件。
- prepared 总大小约 1.7 GB。
- receipt 重复执行可复用。

## Numeric loader

使用官方 `build_vla_dataset` 读取 index 0、15,679 和 31,358：

| Tensor | Shape |
|---|---|
| arm state | `[7]` |
| gripper state | `[1]` |
| arm action chunk | `[50, 7]` |
| gripper action chunk | `[50, 1]` |

三个位置 task 文本一致，所有 tensor finite。

## Normalization

全量统计 count：

| Feature | Count |
|---|---:|
| `observation.state.arm.position` | 31,359 |
| `observation.state.effector.position` | 31,359 |
| `action.arm.position` | 1,567,950 |
| `action.effector.position` | 1,567,950 |

关键尺度：

- Arm delta mean 接近 0，各维 std 约 0.059-0.257。
- Gripper state std 约 0.435。
- 所有 mean/std/min/max/quantile finite。

## Full training sample

Index 0 的完整处理结果：

| Tensor | Shape / Result |
|---|---|
| processed images | `[3, 256, 1536]` |
| current RGB teacher images | `[3, 3, 256, 256]` |
| future RGB teacher images | `[3, 3, 256, 256]` |
| image grid | `[3, 3]` |
| state | `[55]` |
| actions | `[50, 55]` |
| joint mask | `[50, 55]` |
| language tokens | `[72]`, 25 active |
| active state dims | 8 |
| active action dims | 8 |
| normalized state abs max | 0.886 |
| normalized action abs max | 0.910 |

当前和未来图像标准差均约 69.7，确认不是空白帧；所有输出 finite。

## Camera wiring audit

2026-07-18 对同一帧的四路视频进行并排视觉核验。采集时的 raw 名字与物理安装位不一致：

| raw key | 画面可验证的实际角色 |
|---|---|
| `left_eye` | head/global A |
| `left_wrist` | head/global B |
| `right_eye` | wrist A |
| `right_wrist` | wrist B |

训练输入已按实际画面角色修正为 `camera_top <- left_eye`、两个 wrist slot `<- right_eye/right_wrist`。物理左右仍待厂商线序表或逐路遮挡实验确认；在此之前 raw key 作为稳定 stream ID，训练和部署必须使用同一映射。

## GPU training smoke

2026-07-18 在 8 张 NVIDIA H200 上完成真实 2-step FSDP2 smoke：

| 指标 | Step 1 | Step 2 |
|---|---:|---:|
| total loss | 0.8694 | 0.6284 |
| VLA loss | 0.8350 | 0.5972 |
| depth loss | 4.1542 | 3.7060 |
| future depth loss | 3.8223 | 3.4943 |
| future video loss | 0.3345 | 0.2272 |
| grad norm | 5.7266 | 2.3336 |

- Global batch size：8（micro batch 1 x 8 ranks）。
- 模型参数：6,375.9M；每步 activated 参数约 5,186.8M。
- 峰值单卡显存：约 27.37 GB。
- 所有 loss、梯度和输入 tensor finite。
- DCP checkpoint 和 6-shard HF checkpoint 均成功保存。
- 训练进程退出码：0。
- Smoke checkpoint 总大小：约 93 GB，位于代码仓库之外。

实跑同时发现并修正两个上游配置兼容点：v2 policy 应使用 `freeze_vision_encoder` 而不是 legacy `freeze_vit`，启用 alignment 时必须提供 `align_params.visual_steps`。两项都已加入模板和回归测试。

## 100-step stability run

两步 smoke 通过后，又从原始 6B checkpoint 独立执行了 100-step run，退出码为 0，并成功保存 step 100 的 DCP/HF checkpoint。为避免单 batch 波动造成误判，比较前后各 20 步均值：

| 指标 | Steps 1-20 | Steps 81-100 |
|---|---:|---:|
| total loss | 0.5538 | 0.3291 |
| VLA loss | 0.5315 | 0.3188 |
| depth loss | 2.6616 | 1.1449 |
| future depth loss | 2.4919 | 1.1249 |
| future video loss | 0.1189 | 0.0206 |
| grad norm | 2.2029 | 1.7529 |

100 步内 VLA loss 范围为 `0.1682-1.0562`，符合不同 episode/action chunk 的 batch 波动；未出现 NaN/Inf、非零 `Ignore_Batch_Num`、rank 退出或显存增长。该 run 使用 100-step cosine schedule，只作为稳定性验证，不作为正式训练 checkpoint 续跑来源。

## 2000-step formal run

2026-07-18 在 8 张 NVIDIA H200 上从公开 6B checkpoint 独立执行正式 2000-step 微调。global batch size 为 8，checkpoint 间隔为 500 steps。

| Window | Total loss | VLA loss | Depth loss | Future depth | Future video | Grad norm |
|---|---:|---:|---:|---:|---:|---:|
| steps 1-50 | 0.5469 | 0.5236 | 2.7937 | 2.6176 | 0.1379 | 2.6414 |
| steps 451-500 | 0.2849 | 0.2780 | 0.6948 | 0.6990 | 0.0096 | 1.6830 |
| steps 951-1000 | 0.2825 | 0.2764 | 0.6060 | 0.6055 | 0.0078 | 1.6147 |
| steps 1451-1500 | 0.2530 | 0.2472 | 0.5617 | 0.5674 | 0.0072 | 1.5616 |
| steps 1951-2000 | 0.2557 | 0.2499 | 0.5494 | 0.5587 | 0.0070 | 1.6233 |

- 达到 `2000/2000` 后按预期退出，shell/torchrun ranks 均正常结束。
- 全程没有 NaN/Inf、非零 `Ignore_Batch_Num`、Traceback 或 elastic failure。
- `global_step_500/1000/1500/2000` 均包含完整 DCP state 和 6-shard HF checkpoint。
- 每个 checkpoint 约 92 GiB，全部保存在代码仓库之外。
- 后半程受到同机其他 GPU 作业竞争，step time 从约 2.1 秒上升到约 3.7 秒，但没有影响数值稳定性。

结束摘要中的 `eval_failed=5` 不是训练或权重转换失败。该版本 saver 会把成功保存但没有提交/跳过自动评测的记录计入此字段；同一摘要显示 `hf_success=5`、`hf_failed=0`、`failures=[]`。本项目在训练结束后单独运行 open-loop evaluation。

## Open-loop inference

使用 episodes `0, 10, 20, 30, 43`，每条执行 3 个 50-step chunks，共比较 750 个 unnormalized action positions：

| Checkpoint | Average MSE | Average MAE |
|---|---:|---:|
| step 1500 | 0.008946 | 0.057300 |
| step 2000 | **0.007354** | **0.051602** |

两个 checkpoint 均成功完成模型加载、三路视频解码、Qwen3-VL 预处理、10-step denoising 和 8D 动作反变换。step 2000 在全部 5 条测试 trajectories 上均取得更低误差，是当前 replay 协议下的首选。稳定推理耗时约 0.66 秒/50-step chunk；详细边界和真机测试要求见 [EVALUATION.md](EVALUATION.md)。

## 自动测试

单元测试覆盖：

- contract revision 和必要字段。
- 合法 fixture audit。
- 被动 action 关系破坏时 hard fail。
- 2%-5% timestamp FPS 偏差只产生 warning。
- layout acceptance hash。
- prepared receipt 身份和 contract hash。
- prepared metadata 结构和 feature width。
- norm stats SHA256 篡改检测。
- template 缺失变量。
- stale acceptance 拒绝。

## 剩余验证

- 构建独立 held-out episodes；当前 open-loop 指标来自 training replay，不能解释为泛化性能。
- 完成真实机器人 shadow test 和受限速度 rollout，并用任务成功率最终选择 checkpoint。
- 在部署端确认 7 个右臂关节的顺序、单位和 delta 反变换。
- 在重接相机线或修改数据 key 前，确认四路相机物理左右并做版本化迁移。
