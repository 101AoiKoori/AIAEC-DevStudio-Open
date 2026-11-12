# 实验区

识别任务的入口为 [`features/README.md`](features/README.md)。每项识别任务分别建立训练和推理实验目录，推理核心直接编写在 `ws/src/aec_vision/` 并供实验与 ROS 共用；详见 [`docs/PERCEPTION_DEVELOPMENT.md`](../docs/PERCEPTION_DEVELOPMENT.md)。实验时记录样本、算法版本、评估指标、失败案例和实车结论。`navigation/` 用于导航实验，`integration/` 用于联调。

`training_common/` 提供各任务共享的训练执行、独立评估、有限调参、失败分析和图表输出，任务入口与配置分别在 `features/<任务>/training/scripts/`、`config/`。`training_common/portable_env/` 提供 Windows 独立依赖构建、兼容检查和隔离启动；其中 `bundles/` 是可复制第三方依赖产物，`runs/` 保存环境构建与验证记录，均不提交 Git。工作空间训练输出统一到根目录 `runs/`；额外分发镜像由 `tools/build_training_packages.py` 生成到 `training_packages/`，不能反向成为实验区的代码或数据依赖。
