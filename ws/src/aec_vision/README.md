# 共用识别核心

这里存放独立实验和 ROS 适配层共同导入的推理模块，按 `aec_vision/<任务ID>.py` 命名，每个任务按需扩展为包。核心独立于 ROS 和实验目录；当前实现与验证范围见[识别开发设计](../../../docs/PERCEPTION_DEVELOPMENT.md#共用推理脚本)。

在源码仓库进行实验时，可将本包的 `src/` 加入 `PYTHONPATH` 后导入；在 Ubuntu 运行 `catkin_make` 并 source 工作空间后，ROS 适配层通过同一个 Python 包导入。共享的是同一份源码，实验和部署各自运行在自己的系统环境中。

任务开关、候选路线与 waypoint 配置见 `config/competition.json`，读取和检查组件位于 `aec_vision/runtime_config.py`。它不会自行启动任务或控制导航，操作及边界见仓库的 `docs/TASK_SWITCHES_AND_ROUTES.md`。
