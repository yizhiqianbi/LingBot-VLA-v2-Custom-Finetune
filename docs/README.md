# 文档索引

本目录是 LingBot-VLA-v2 自采数据微调工程的统一文档入口。文档按数据、工作流、训练、评测和验证记录拆分，避免同一状态或命令在多个文件中维护。

## 名词边界

- **LingBot-VLA-v2**：本仓库微调的 6B vision-language-action 模型。
- **LingBot-VA data**：其他数据管线中的上游数据源名称，不等于本模型，也不使用本仓库的 55D 合同。
- **Source dataset**：Hugging Face 上只读的 LeRobot v2.1 数据。
- **Prepared dataset**：非破坏转换后的 LeRobot v3.0 训练副本。
- **DCP checkpoint**：恢复训练所需的 model/optimizer state。
- **HF checkpoint**：`hf_ckpt/` 下用于推理的 safetensors。

FastWAM 与 LingBot-VLA-v2 可以读取同一批 raw robot data，但不能混用 canonical 格式、normalization stats、action mask 或 checkpoint。

## 建议阅读路径

复现训练：

1. [当前验收状态](reference/VALIDATION_STATUS.md)
2. [端到端工作流](workflow/PIPELINE.md)
3. [数据与映射](data/DATASET_AND_MAPPING.md)
4. [微调指南](training/FINETUNING.md)
5. [评测与部署](evaluation/EVALUATION_AND_DEPLOYMENT.md)

继续下一阶段实验：

1. [执行路线图](ROADMAP.md)
2. [当前验收状态](reference/VALIDATION_STATUS.md)
3. [评测与部署](evaluation/EVALUATION_AND_DEPLOYMENT.md)

## 文档职责

| 文档 | 唯一职责 |
|---|---|
| [Roadmap](ROADMAP.md) | 下一步执行顺序、阶段门槛和完成定义 |
| [Validation Status](reference/VALIDATION_STATUS.md) | 已实际运行并获得证据的结果 |
| [Dataset and Mapping](data/DATASET_AND_MAPPING.md) | 数据身份、action/state 语义、相机错线和 owner acceptance |
| [Pipeline](workflow/PIPELINE.md) | 下载到训练启动前的完整数据与配置流程 |
| [Fine-tuning](training/FINETUNING.md) | 训练参数、实验阶梯、checkpoint 恢复和故障排查 |
| [Evaluation and Deployment](evaluation/EVALUATION_AND_DEPLOYMENT.md) | open-loop、shadow test、动作安全和 closed-loop 协议 |

## 单一事实来源

- “已经跑通什么”只更新 `reference/VALIDATION_STATUS.md`。
- “接下来先做什么”只更新 `ROADMAP.md`。
- 数据 revision、维度语义和相机映射只维护在 contract 与 `data/DATASET_AND_MAPPING.md`。
- 下载、转换、receipt 和 normalization 流程只维护在 `workflow/PIPELINE.md`。
- 训练参数与恢复规则只维护在 `training/FINETUNING.md`。
- replay、shadow 和真机协议只维护在 `evaluation/EVALUATION_AND_DEPLOYMENT.md`。

当实现、配置和文档冲突时，版本化 contract、模板和测试是最终依据；文档应在同一提交中修正。

## 状态用语

- **Implemented**：代码路径存在，尚不表示真实数据运行成功。
- **Data-validated**：固定 revision 的数据审计、转换和 loader 验证通过。
- **Training-validated**：完成 optimizer step、checkpoint 保存和 resume 验证。
- **Replay-validated**：记录数据上的 open-loop 推理通过，不代表泛化。
- **Deployment-ready**：held-out、shadow test、安全门槛和受限 closed-loop 全部通过。

文档中不得用笼统的“已调通”替代这些状态。
