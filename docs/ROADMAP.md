# LingBot-VLA-v2 下一阶段路线图

更新日期：2026-07-19

本文只维护下一步顺序、阶段门槛和完成定义。已获得的真实结果见 [Validation Status](reference/VALIDATION_STATUS.md)，具体命令由工作流、训练和评测文档维护。

## 目标

把已经完成的 2000-step 训练集拟合基线升级为可选择、可部署、可复现的目标任务模型：

1. 建立 episode-level held-out 数据集和 train-only normalization stats。
2. 用独立 validation 选择 checkpoint，而不是依赖 training loss 或 replay。
3. 完成真机 shadow test、动作安全审计和低速 closed-loop。
4. 固化相机修线前后的 stream mapping，避免训练与部署输入错位。

## 当前边界

| 模块 | 已完成 | 尚未完成 |
|---|---|---|
| 数据 | 44 episodes 全量审计、v2.1 -> v3.0、三路相机和 8D action mapping | episode-level held-out revision/subset |
| 训练 | 2-step smoke、100-step stability、2000-step 8-GPU run、DCP/HF checkpoints | held-out 重训和独立 checkpoint selection |
| 推理 | step 1500/2000 训练集 replay | held-out open-loop |
| 部署 | 画面角色已核验，训练映射已固定 | 关节顺序/单位确认、shadow test、closed-loop |

当前 step 2000 是 replay 协议下的候选 checkpoint，不是最终部署模型。

## P0：冻结现有基线

开始新实验前保存：

- 本仓库 commit、`upstream.lock` 和 dataset revision。
- 当前 contract、layout acceptance、prepare receipt 和 norm manifest。
- step 2000 的 DCP/HF checkpoint 完整性报告。
- 2000-step resolved config、命令、环境和 replay 结果。
- 相机 raw stream 到画面角色的当前映射。

**退出门槛**：从固定 commit 和外部 artifact 路径可以重放一个相同输入 chunk，并得到 finite 8D action。

## P1：建立 Held-out 数据合同

推荐按完整 episode 划分：

```text
train: 36-40 episodes
validation: 4-8 episodes
split unit: episode or collection condition
```

必须完成：

- 生成明确的 train/validation episode 清单和哈希。
- 保证同一采集序列的相邻帧不跨 split。
- 为 train subset 建立独立 prepared receipt 和 runtime manifest。
- 只用 train subset 重新计算 normalization stats。
- validation loader 复用相同 feature mapping，但不写入训练统计。
- 记录各 split 的帧数、时长、初始物体位置和画面分布。

禁止从全量 norm stats 中简单删除 validation 行，也禁止按 frame 随机划分。

**退出门槛**：train/validation 数据、receipt、stats 和 manifest 可由脚本重新生成，且没有 episode/lineage 泄漏。

## P2：Held-out 微调与模型选择

实验顺序：

1. 在新 split 上运行 numeric/full-sample smoke。
2. 运行 2-step GPU smoke 并验证 DCP/HF 保存。
3. 先执行短 overfit，确认新 train subset 的 loss 可下降。
4. 按固定间隔保存正式微调 checkpoint。
5. 在完全相同的 held-out chunks 上比较所有候选 checkpoint。

至少报告：

- normalized 和 unnormalized arm/gripper MSE、MAE。
- 每个 action dimension 的曲线和误差分布。
- 50-step horizon 内误差随时间的变化。
- 推理延迟、失败样本和相机/视频解码错误。
- training 与 validation 指标差距。

**退出门槛**：候选 checkpoint 由 held-out 指标选出，且能从 DCP 恢复训练、从 HF 权重独立推理。

## P3：Shadow Test

模型动作只记录、不下发控制。逐项确认：

- 三路图像按训练时的 raw stream ID 输入，不按错误 key 名重排。
- 7 个右臂关节顺序、单位、正负方向与控制器一致。
- arm delta 使用 `current_state + predicted_delta` 恢复绝对目标。
- gripper 保持 absolute target 语义。
- `[7:14]` 未训练通道保持当前状态，不被模型覆盖。
- joint position、velocity、acceleration、jerk 和 workspace 均满足硬限制。
- 连续预测没有突跳、累积漂移或夹爪方向反转。

**退出门槛**：固定回放和在线观测均通过动作安全检查，没有 mapping mismatch。

## P4：受限 Closed-loop

从低速、缩小 workspace、急停可用的条件开始。模型生成 50-step chunk，但初期只执行前 1-4 步后重新规划，并在控制器侧限幅。

每个 checkpoint 使用相同初始条件，记录：

- 完整任务成功率。
- 正确抓取和正确分类放置率。
- 掉落、碰撞、错误抓取次数。
- 人工干预和急停率。
- 完成时间、重规划次数和动作延迟。

**退出门槛**：目标 checkpoint 在预定义试验次数上满足成功率和安全阈值，结果可由日志和视频复核。

## 相机修线迁移

修线或重命名 raw key 时不能直接修改训练 mapping。应建立新的 mapping version，并完成：

1. 四路同步遮挡或标定实验，确认物理左右。
2. 保存旧 stream ID -> 画面角色 -> 新 stream ID 对照表。
3. 对同一时刻生成旧/新 pipeline 输入并做像素角色检查。
4. 重新生成 layout acceptance、runtime config 和部署配置。
5. 运行 full-sample、replay 和 shadow regression。

## 实验记录

每次实验使用新的外部输出目录，并至少保存：

```text
command.txt
resolved_config.yaml
code_commits.json
dataset_split.json
prepare_receipt.json
normalization_manifest.json
environment.txt
metrics.jsonl
checkpoints/
evaluation/
```

## 停止条件

出现以下任意情况立即停止训练或部署：

- loss、gradient 或 action 出现 NaN/Inf。
- joint mask、active dimensions 或 action 反变换不一致。
- 相机画面角色与训练 mapping 不一致。
- checkpoint 不完整或 DCP/HF 用途混淆。
- held-out 指标持续恶化而 training loss 继续下降。
- shadow action 超出硬件限位或出现方向/单位错误。

## 完成定义

- [ ] held-out split、train-only stats 和无泄漏报告完成。
- [ ] held-out 微调、checkpoint comparison 和独立 HF 推理完成。
- [ ] 关节顺序、单位、delta/absolute 语义由部署端确认。
- [ ] shadow test 通过全部动作安全门槛。
- [ ] 受限 closed-loop 有可复现成功率和干预率。
- [ ] 相机映射在修线前后有版本化迁移记录。

具体已完成项不要在本文重复更新，统一写入 [Validation Status](reference/VALIDATION_STATUS.md)。
