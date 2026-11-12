# 正式 catkin 工作空间

`src/` 是部署区。`aec_vision` 提供实验与 ROS 适配层共用的识别核心，当前实现与验证范围见[识别开发设计](../docs/PERCEPTION_DEVELOPMENT.md#共用推理脚本)。ROS 节点、实车 topic、action、相机及红外接口仍待实现和验证。

项目自建包采用 `aec_<职责>` 格式，现有 `aec_vision` 负责识别；其他职责是否拆包以实际需求为准。命名约定见 [项目方向](../docs/PROJECT_DIRECTION.md)。省赛 `bag` 包可逐文件迁移，不应整体覆盖这里。

正式包只接入推理必需的代码、参数、模型和 ROS 适配层，不导入训练脚本或实验目录。实验和 ROS 共用包内源码；模型单独传送到部署包约定的位置。训练、推理及文档流程见 [`docs/PERCEPTION_DEVELOPMENT.md`](../docs/PERCEPTION_DEVELOPMENT.md)。
