# 开发约定

在实际 checkout 内处理用户指定的任务。数据制作进度在[当前生产进度](docs/DATASET_PRODUCTION.md#当前生产进度)维护。数据集脚本的调用方式见[脚本入口](datasets/scripts/README.md)。

视觉模型研发从学习目标和组合轴经数据制作、训练与微调到交付 `.pt`；总体能力与边界见[视觉研发架构](docs/VISION_ARCHITECTURE.md)。

## 小车与部署

- 正式运行目标为 Ubuntu 20.04 + ROS Noetic + Python 3。Windows 检查记为离线结果。
- `ws/src/` 只放可部署 catkin 包。正式运行使用部署包和目标机已安装的依赖。
- 导航使用 ROS `move_base` 和国赛现场标定的地图、坐标。
- 按可靠的禁行牌结果选择路线。无法可靠识别时停车并记录故障。
- 红灯停在等待线前；碰撞、压线、超时均是比赛风险。速度、重试、超时要有明确上限。
- 按国赛规则输出识别结果、图片标注和终端播报。失败时输出实际失败状态。
- 新增相机或红外设备前核对传感器数量限制，记录型号、话题、分辨率与标定结果。
- 不提交原始数据、模型权重、密钥、虚拟环境及 catkin 生成目录。

共享推理核心位于 `ws/src/aec_vision/src/aec_vision/`，实验和 ROS 适配层导入同一模块。自建 ROS 包采用 `aec_<职责>`，包目录、`package.xml`、CMake 项目名及包内导入一致。实车交付、模型装载和部署变更按 `workspace-offline-delivery` 处理。

## 数据设计与实现

V4 的八项数据任务见[协作协议](datasets/DATASET_COORDINATION.md#工作对象和落点)，车牌沿用省赛模型。各任务有独立定义与素材入口，共享同一套 Skill；学习目标与组合轴分别维护于 `config/task_targets.yaml` 和 `config/combination_axes.yaml`。

**以任务识别成功为目标，先用可行实拍、合成和 Blender 形成训练数据并推进模型。** 不把缺少某类实拍、无法预先确定效果、设计比例或人工签收作为普遍停止条件；每模型安排一次基线，后续按实际失败和用户指定预算补数据、微调。来源、复用及评估范围如实记录；独立真实场景结论使用相应证据。制作和入库细则使用 `dataset-production-loop`。

**禁止未声明的否决。** 全局执行约定由本文件维护，识别需求、监督目标、组合轴、目录及任务参数分别由其现有配置维护。每项业务规则只有一个可写入口。文档解释或引用，Skill 按当前任务调用和处理，脚本执行，测试验证。历史记录、证据未知和 Agent 临场判断本身没有否决权。遇到拒绝，给出实际失败、权威键及路径、当前值与要求值、影响范围和最小解除方法；无授权来源的限制按实现缺陷处理。实际缺文件、输入无效和写入冲突报告具体技术原因。

**正式版本由用户决定发布。** 授权修改当前数据或配置时，更新实际维护源及本次受影响的数据和派生内容。执行回执、数据库证据修订和运行快照用于追溯；创建发布标签、正式数据版本或版本目录须有用户对该次发布的明确授权。

按用途保留用户原件、模型权重、V3 记录、持久回执及人工内容。确认小样或导出有误时，清理相应 `datasets/samples/<任务>` 或 `datasets/exports/<任务>/<split>` 的生成结果并重跑，保留目录、原件及人工内容。清理文档时仅处理文档；只验证受影响的实际入口及相关契约。

## 修改项目文件

- 识别需求改 `config/perception_tasks.json`，再运行 `python lab/scripts/contracts/sync_docs.py` 更新派生文档。目录契约改 `config/workspace_layout.yaml`，用 `sync_layout_docs.py` 更新生成区块并用 `workspace_layout.py check` 检查布局；这两个脚本在 `lab/scripts/contracts/`。
- 分类学改 `config/workspace_ontology.yaml`，编号和状态定义改 `config/id_policy.yaml`。工作空间级工具归 `lab/scripts/` 的相应职责目录。修改公共接口前定位调用方，在已有入口验证受影响行为。
- Blender 源资产登记库、可重建媒体查询缓存和持久生产回执各有用途，具体路径见[数据区说明](datasets/README.md)。变更素材引用时更新受影响的记录；原始素材、外来文件、既有产物的删除或迁移以该操作获得的授权为准。
- 外部源码须核实来源及许可后引入本项目；目标工程脚本只加载项目内源码或已安装的工具与依赖。归档和恢复使用 Git 与 `workspace-history-archive` 约定。
- 独立阶段完成后检查并提交该阶段的变更。共享工作树和索引由一个写者串行操作，保留其他人的工作。

## 任务开始时的 Skill 分流

推进或接续视觉任务从学习目标、组合轴、数据制作到训练和 `.pt` 模型的工作，以及定位其中的执行断点，先使用 `vision-model-workflow`；按当前阶段再调用相关专项 Skill。视觉研发的交付边界是模型，其他协作方负责实车任务流程。

选与眼前工作直接对应的入口。数据集制作使用 `dataset-production-loop`；查询或修订文档使用 `workspace-doc-governance`；目录归属用 `workspace-file-governance`；接口用 `shared-code-interfaces`；实验数据与指标用 `workspace-experiment-governance`。需要跨窗口协作时查[窗口通信与共享写入](datasets/DATASET_COORDINATION.md#窗口通信与共享写入)。普通本地编辑可以在当前窗口完成。Skill 是按需调用的工作指导，具体执行以实际配置、脚本和产物为准。

 查询、补证、分配或修订当前逐图记录、数据配置、识别需求或组合轴时，用 [dataset-config-operations](.agents/skills/dataset-config-operations/SKILL.md) 找到维护源和实际消费者。用户说“同步某任务的当前数据”时，核对当前产物及其调用入口，修复本次范围内的接线和引用。

制作中切换计划、split、清单路径或字段含义时，在同轮调用 `dataset-config-operations` 同步当前计划登记、实际消费者和派生文档；可复用的接线经验按该 Skill 回写对应现有 Skill。只读核查不触发写入，运行中的批次输入保持其生成时记录。
