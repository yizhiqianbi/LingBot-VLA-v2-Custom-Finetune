# Dataset and Feature Mapping

## 1. 数据集身份

| 字段 | 值 |
|---|---|
| HF repo | `jokeru/take_wrong_item_right_arm` |
| 固定 revision | `76a008f11dc65a99d176061fb59807a9abdfdffc` |
| 原始格式 | LeRobot v2.1 |
| episodes | 44 |
| frames | 31,359 |
| nominal FPS | 28 |
| task 数 | 1 |
| 原始 state/action | 15D / 15D float32 |
| 视频 | H.264, 744x960, 28 FPS |

Task：

> Use the right arm to place the misplaced object on the shelf into the location of its correct category.

原始 `features.observation.state.names` 和 `features.action.names` 都是 `null`。任何关节语义都不能仅由 metadata 证明。

## 2. 摄像头

原始数据包含四路同步视频：

- `observation.images.left_eye`
- `observation.images.right_eye`
- `observation.images.left_wrist`
- `observation.images.right_wrist`

LingBot-VLA-v2 公开训练配置使用三个标准视角，因此当前映射为：

```yaml
camera_top: observation.images.right_eye
camera_wrist_left: observation.images.left_wrist
camera_wrist_right: observation.images.right_wrist
```

选择 `right_eye` 作为 global camera 的理由是任务明确限定 right arm。保留两路 wrist camera 是为了保持公开模型的三视角结构和位置先验。`left_eye` 不送入模型，但源审计仍检查其文件完整性。

## 3. 动作数值审计

审计遍历了全部 44 个 parquet，而不是抽样。对每个 episode 内所有非 terminal frame，计算：

```text
abs(action[t, 7:15] - state[t+1, 7:15])
```

结果：最大误差为 0。相同检查在 `0:7` 上的平均误差约为 0.006542，最大误差约为 0.477723，因此 `0:7` 不是简单复制下一帧观测，而更符合独立控制目标的特征。

当前解释：

- `0:7`：任务右臂的 7 个独立控制关节。
- `7:14`：非任务臂/被动通道，action 由下一帧 state 填充，不应作为本任务 policy 输出。
- `14`：右夹爪目标。它同样等于下一帧 state，但夹爪绝对目标采用这种记录方式是合理的。

第 14 维 state 范围约为 `[-0.0330, 1.2457]`，也与单自由度夹爪信号一致。

## 4. 统一特征

Robot config 将原始数组映射到 LingBot 的统一命名空间：

```yaml
states:
  - observation.state.arm.position: observation.state[0:7]
  - observation.state.effector.position: observation.state[14:15]

actions:
  - action.arm.position: action[0:7]
    subtract_state: true
    relative_type: joint
  - action.effector.position: action[14:15]
    subtract_state: false
```

`relative_type: joint` 必须显式指定。官方 `FeatureTransform` 对缺省 relative type 使用 `quaternion_local`；若只写 `subtract_state: true`，7D 关节会错误进入末端姿态四元数分支。

最终有效 state/action 均为 8D：

```text
7D arm + 1D gripper = 8D active
```

LingBot 训练 head 保持公开 checkpoint 的 55D 结构。Loader 按训练配置先给 feature 类别分配 padding，再 pad 到 55D，并通过以下 mask 保证 padding 不参与损失：

- `state_joint_mask`: 55D，其中 8 个 True。
- `action_joint_mask`: 55D，其中 8 个 True。
- `joint_mask`: `50 x 55`，每个 action step 仅 8 个 active dims。

## 5. 动作表达

右臂动作使用：

```text
delta_arm[t+k] = raw_action[t+k, 0:7] - state[t, 0:7]
```

注意，整个 50-step chunk 都相对于当前时刻 state，而不是每一步分别减去未来 state。这与官方 `FeatureTransform` 的训练和部署反变换一致。

夹爪动作保持绝对值：

```text
gripper[t+k] = raw_action[t+k, 14]
```

Chunk size 为 50，在 28 FPS 下覆盖约 1.79 秒。公开模型的 future frame offset 是 `(50 - 1) / 28 = 1.75` 秒。

## 6. 归一化

四个 feature 均使用 `meanstd`：

- `observation.state.arm.position`
- `observation.state.effector.position`
- `action.arm.position`
- `action.effector.position`

统计针对映射和 delta 处理后的数据计算，而不是直接对原始 15D 数组计算。Action 统计覆盖 Loader 实际产生的全部 50-step chunks，包括 episode 尾部按官方逻辑 padding 的位置，因此与训练分布严格一致。

统计输出采用官方 `lingbotvla.utils.normalize.save` 格式，同时生成独立 manifest，记录 count、SHA256、chunk size、数据 revision、upstream revision 和 layout confirmation 状态。

## 7. 时间戳质量

44 个 episode 内时间戳均严格单调，段内步长几乎恒定。各段由时间戳估计的 FPS 范围约为 27.829 到 28.615，中位数约 28.384。

Episode 11 为 28.615 FPS，与 nominal 28 FPS 相差 2.20%。它低于 5% 的 hard-fail 阈值，因此作为 warning 保留。视频 metadata 和编码 FPS 仍为 28，未发现视频缺失或首帧解码失败。

## 8. 必须由数据所有者确认

正式 acceptance 前请逐项核对：

1. 原始 `0:7` 的顺序确实是部署侧右臂 7 个关节顺序。
2. 原始 `14` 确实是右夹爪，而不是其他本体自由度。
3. `7:14` 不应由本任务 policy 控制。
4. `right_eye` 确实是适合右臂任务的 global camera。
5. 部署侧执行动作时，会对 arm delta 做 `state + delta`，对 gripper 保持 absolute。

代码不会自动替数据所有者做这项确认。
