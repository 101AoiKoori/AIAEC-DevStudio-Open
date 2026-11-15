# 实验区

识别任务的入口为 [`features/README.md`](features/README.md)。每项识别任务分别建立训练和推理实验目录，推理核心直接编写在 `ws/src/aec_vision/` 并供实验与 ROS 共用；详见 [`docs/PERCEPTION_DEVELOPMENT.md`](../docs/PERCEPTION_DEVELOPMENT.md)。实验时记录样本、算法版本、评估指标、失败案例和实车结论。`navigation/` 用于导航实验，`integration/` 用于联调。

[`scripts/`](scripts/README.md) 是工作空间级脚本区：`contracts/` 负责目录契约与生成文档，`media/` 负责索引与抽帧，`blender/` 负责桥接与资产登记，`data/` 负责来源门控、打包与评估，`audit/` 负责生产入口审计，`common/` 负责人工辅助标注与来源审查。这里不是任务生产入口或正式运行依赖，也不保存数据产物；任务专用生成入口仍在 `datasets/scripts/<任务>/`。

`training_common/` 提供各任务共享的训练执行、独立评估、有限调参、失败分析和图表输出，任务入口与配置分别在 `features/<任务>/training/scripts/`、`config/`。`training_common/portable_env/` 提供 Windows 独立依赖构建、兼容检查和隔离启动；其中 `bundles/` 是可复制第三方依赖产物，`runs/` 保存环境构建与验证记录，均不提交 Git。工作空间训练输出统一到根目录 `runs/`；`training_packages/` 只保留 `portable_env/`，任务包生成入口已于 2026-11-15 删除，不能再作为实验区的代码或数据依赖。
