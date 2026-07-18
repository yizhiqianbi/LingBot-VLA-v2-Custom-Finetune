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

## 未完成项

当前执行环境 `torch.cuda.is_available()` 为 false，因此无法完成：

- 公共 6B checkpoint 的 GPU 构建。
- MoE fused kernel 编译。
- forward/backward/optimizer step。
- 两步 distributed train smoke。

这些步骤已经由 `scripts/check_environment.sh --require-cuda` 和 `scripts/train_smoke.sh` 封装，必须在有兼容 CUDA GPU 的节点继续执行。数据、归一化和完整 sample 链路已经调通。
