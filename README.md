# AIAEC 国赛小车技术与开发文档

本仓库提供项目当前源码、配置和技术文档，供技术文档编写及协作使用。原始采集、数据集、模型权重、训练包、运行结果和历史版本未包含；部分技术文档指向本地素材、历史产物或比赛与设备原件，这些引用需要另行获得授权后提供。`datasets/.gitkeep` 用于保留数据目录，不提供数据；需要时自行建立 `raw/`、`annotated/` 和 `exports/` 子目录。

目标平台：Ubuntu 20.04、ROS Noetic、Python 3、catkin。Windows 侧用于编辑、数据整理和离线验证；现场通过 NoMachine 进入小车 Ubuntu 桌面，在小车本机终端构建和启动。

## 目录

| 路径 | 用途 |
| --- | --- |
| `.agents/skills/` | 本工作空间的 Agent 技能与其直接使用的脚本 |
| `.codex/agents/` | 本项目可复用的 Codex 专职子代理配置 |
| `.githooks/` | 仓库 Git 钩子；启用方式由本机 Git 配置决定 |
| `config/` | 跨任务需求配置；识别任务需求以其中的 `perception_tasks.json` 为源 |
| `ws/src/` | 正式 ROS 包开发区，最终部署代码只从这里构建 |
| `lab/features/` | 各识别功能的独立实验，不能被正式包直接导入 |
| `lab/navigation/` | 建图、路线、禁行分支、泊车实验 |
| `lab/integration/` | 多任务联调与计时实验 |
| `datasets/` | 采集、标注、评估数据；原始数据不入 Git |
| `training_packages/` | 可按分发需要替换或重建的训练包，含脚本、数据和权重，不入 Git |
| `lab/training_common/` | 各任务共用训练、评估、调参、可视化和 V1/V2 试验工具；`portable_env/` 管理 Windows 独立依赖包的构建与启动 |
| `runs/` | 本机训练、评估、调参的完整运行产物；按任务、版本、运行编号管理，不入 Git |
| `models/` | 模型权重与版本记录；大文件不入 Git |
| `docs/` | 任务矩阵、接口调查、现场流程、技术文档 |
| `refs/provincial/` | 省赛迁移笔记；原仓库保持独立，不复制旧赛道 |
| `refs/archive/` | 已被当前方案替代的项目研究记录，仅供历史复核，不参与默认文档检索 |
| `refs/` | 项目参考资料与迁移笔记；非正式运行依赖 |
| `Archive/` | 外来原件及 `technical/` 国赛工作空间技术迭代归档；保留来源，不作为正式运行依赖，文档历史不参与技术归档 |
| `tools/` | 仓库级维护工具及其检查，非产品运行时模块 |
| `tmp/` | 本机可丢弃的临时产物，不提交；不可作为唯一数据或正式运行依赖 |
| `国赛规则说明.pdf` | 比赛规则原件未随仓库分发；获取方式和再分发许可需另行确认 |

### 文件归属规则

创建文件前先判断用途和来源：部署代码、配置与包内测试放对应 `ws/src/aec_<职责>/`；按识别任务组织的训练、推理实验及其测试放 `lab/features/<任务>/`，导航和联调实验分别放 `lab/navigation/`、`lab/integration/`。项目分析与技术决策放 `docs/`，外部资料及其来源说明放 `refs/`；现有 `Archive/` 内原件保留原位，新引入外部资料前记录来源、版本、许可和用途，无法确认时标记待核实。数据和模型分别遵循 [`datasets/README.md`](datasets/README.md) 与 [`models/README.md`](models/README.md)，不把原始数据、权重或凭据提交 Git。

临时结果放 `tmp/` 或已有忽略的实验输出目录；需要长期保留的结果应转为有来源、适用范围和验证状态的项目记录，而不是把临时目录当资料库。新增目录需明确职责并更新本表或所属目录说明。移动现有文件前检查代码、配置、文档及生成器中的引用，移动后同步修复；不因整理而擅自处理未识别文件、原始数据、凭据或外部原件。

路径自动检查可运行 `python .agents/skills/workspace-file-governance/scripts/check_layout.py`；用 `--path 相对路径` 检查拟使用的路径，或用 `--suggest training-script --task meter` 等按用途建议目录。启用 `.githooks` 的机器在提交时检查暂存路径；钩子可以被跳过。检查仅覆盖可按路径判断的规则，未知职责提示人工核实，不读取被忽略的数据或凭据内容；具体范围及退出码见文件管理 Skill。

## 先做什么

1. 阅读 [任务矩阵](docs/TASK_MATRIX.md)、[小车接入](docs/ROBOT_SETUP.md) 和 [双电脑离线交付](docs/OFFLINE_DEPLOY.md)。
2. 通过 NoMachine 登录实车，按接入文档记录 ROS topic、action、相机、雷达、红外设备和现有工作空间；现有 ROS 包是共享推理骨架，实车适配待接口确认后实现。
3. 在 `lab/features/` 离线验证每类识别；与导航协作方确认两条禁行分支、观察点和终点泊车的接口与现场验证。
4. 在 `ws/src/aec_vision/` 编写实验和 ROS 共用的推理功能，验证后接入一键入口。路线坐标必须现场标定，不沿用省赛坐标。

省赛代码位于相邻的 `catkin_ws` 仓库；其 `ws/src/bag` 的 `move_base`、waypoint、人物、车牌、红绿灯实现可作为迁移输入。国赛规则、赛道和硬件有差异，迁移前需逐项复验。

当前工作重点是全部识别任务，入口见 [识别任务工作区](lab/features/README.md)；训练、独立推理与 ROS 集成的目录划分见 [识别开发设计](docs/PERCEPTION_DEVELOPMENT.md)，采样、后台识别与调度方案见 [异步识别设计](docs/ASYNC_PERCEPTION_DESIGN.md)。

项目的现状、工作边界、阶段方向和 ROS 包命名规范见 [项目方向](docs/PROJECT_DIRECTION.md)。

开源项目与识别方法的调研入口是 [逐任务方法](docs/research/OPEN_SOURCE_METHODS.md) 和 [仪表专项分析](docs/research/METER_VISION.md)。缺少国赛物料或小车数据时，按[项目方向](docs/PROJECT_DIRECTION.md#实施顺序与验收)提出具体采集任务，并继续可独立完成的工作。


## 使用与分发

本仓库不包含项目数据、模型和外部资料原件。部分生成与验证命令需自行准备有权使用的素材、权重及运行环境；现有实验结果的来源和适用范围见相关技术文档。项目代码许可尚待权利人确定，第三方依赖及资料的许可分别核查。
