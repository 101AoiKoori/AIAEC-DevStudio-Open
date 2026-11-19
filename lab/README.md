# 实验区

识别任务的入口为 [`features/README.md`](features/README.md)。每项识别任务分别建立训练和推理实验目录，推理核心直接编写在 `ws/src/aec_vision/` 并供实验与 ROS 共用；详见 [`docs/PERCEPTION_DEVELOPMENT.md`](../docs/PERCEPTION_DEVELOPMENT.md)。实验时记录样本、算法版本、评估指标、失败案例和实车结论。`navigation/` 用于导航实验，`integration/` 用于联调。

[`scripts/`](scripts/README.md) 是工作空间级脚本区：`contracts/` 负责目录契约与生成文档，`media/` 负责索引与抽帧，`blender/` 负责桥接与资产登记，`data/` 负责来源门控、打包与评估，`audit/` 负责生产入口审计，`common/` 负责人工辅助标注与来源审查。这里不是任务生产入口或正式运行依赖，也不保存数据产物；任务专用生成入口仍在 `datasets/scripts/<任务>/`。

`training_common/` 提供各任务共享的训练执行、独立评估、有限调参、失败分析和图表输出。Windows 上从仓库根目录运行 `.\lab\train.ps1 person`，加 `-Resume` 继续训练。任务训练 YAML 保存在 `features/<任务>/training/config/`，启动设置在 `training_common/launcher.yaml`。具体用法和便携包构建见[训练工作流](../docs/TRAINING_WORKFLOW.md#v4-任务包构建与启动)。

`training_common/portable_env/` 提供 Windows 独立依赖构建、兼容检查和隔离启动。其中 `bundles/` 是可复制第三方依赖产物，`runs/` 保存环境构建与验证记录，均不提交 Git。工作空间训练输出统一到根目录 `runs/`。
