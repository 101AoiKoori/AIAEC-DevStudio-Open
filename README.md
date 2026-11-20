# AIAEC DevStudio Open：国赛小车技术与开发文档

本仓库同步项目当前源码、配置、任务定义与技术文档，供技术编写及协作使用。原始数据、标注、权重、训练包、运行结果和外部原件不随仓库分发。历史快照保留其原始状态；其中来源提交属于研发仓库，不属于本公开仓库历史。

目标平台：Ubuntu 20.04、ROS Noetic、Python 3、catkin。Windows 侧用于编辑、数据整理和离线验证；现场通过 NoMachine 进入小车 Ubuntu 桌面，在小车本机终端构建和启动。

## 目录

<!-- WORKSPACE_LAYOUT:START -->
| 路径 | 用途 |
| --- | --- |
| `.agents/skills/` | 本工作空间的 Agent 技能与其直接使用的脚本 |
| `.claude/` | 本机 Claude Code 会话与 worktree 运行时目录；不入 Git |
| `.codex/agents/` | 本项目可复用的 Codex 专职子代理配置 |
| `.githooks/` | 仓库 Git 钩子；启用方式由本机 Git 配置决定 |
| `config/` | 跨任务需求配置；识别任务需求以 `perception_tasks.json` 为源，目录契约以 `workspace_layout.yaml` 为源，训练目标与组合轴以 `task_targets.yaml` / `combination_axes.yaml` 为源，编号与计数口径以 `id_policy.yaml` 为源，对象分类学以 `workspace_ontology.yaml` 为源 |
| `ws/src/` | 正式 ROS 包开发区，最终部署代码只从这里构建 |
| `lab/features/` | 各识别功能的独立实验，不能被正式包直接导入 |
| `lab/navigation/` | 建图、路线、禁行分支、泊车实验 |
| `lab/integration/` | 多任务联调与计时实验 |
| `lab/scripts/` | 工作空间级脚本区；按 contracts、media、blender、data、audit、common 分类，规则见 `lab/scripts/README.md` |
| `lab/scripts/contracts/` | 目录契约、生成文档、训练目标/组合轴覆盖、全量计划、对象分类学、编号口径和父子源配对的检查与生成工具 |
| `lab/scripts/media/` | 媒体索引、SQLite 查询缓存、视频抽帧、合成背景收集和站点背景特征更新工具 |
| `lab/scripts/blender/` | Blender MCP 桥接与源资产登记库工具；不把单帧预览当数据集批次 |
| `lab/scripts/data/` | 数据源门控、候选打包、现场采集评估和站点适配工具；不训练模型 |
| `lab/scripts/audit/` | 生产入口静态审计，检查受管写入、转发、计划入口和未分类路径 |
| `lab/scripts/common/` | 全工作空间共用的人工辅助标注、标签重建、完整框选、边缘/孔洞检查和 Blender 机位规划脚本；任务专用入口仍留在 `datasets/scripts/<任务>/` |
| `datasets/` | 来源工作副本、标注候选与任务生产。`raw/` 只放八任务目标和共享背景工作副本，`annotated/` 保存标注与审查候选，`exports/<任务>/` 保存当前集合，`samples/<任务>/` 保存当前小样，`scripts/<任务>/` 保存生产入口。数据不入 Git |
| `datasets/blender_source_registry.sqlite3` | Blender 源资产当前状态、已核实关系及带证据引用的 manifest 声明/未决 claim 的 SQLite 权威记录；按 `datasets/README.md` 和 `lab/scripts/blender/blender_source_registry.py` 维护 |
| `datasets/assets/blender/` | 只存放 `aiaec_dataset.blend`；Scene、Collection 与历史快照规则见 [`datasets/README.md`](datasets/README.md#blender-源资产与派生数据) |
| `training_packages/` | 当前仅保留 `portable_env/`；任务训练包副本已清理，目录不入 Git |
| `lab/training_common/` | 各任务共用训练、评估、调参、可视化、生产守卫和 V1/V2 试验工具；`portable_env/` 管理 Windows 独立依赖包的构建与启动 |
| `runs/` | 本机历史训练、评估和调参运行记录，不入 Git；内层 `runs/` 中的 V3 模型和记录保持原位。V4 当前小样及数据集分别放在 `datasets/samples/` 和 `datasets/exports/`，不在此堆放并列版本 |
| `models/` | 模型权重与版本记录；大文件不入 Git |
| `docs/` | 任务矩阵、接口调查、现场流程、技术文档 |
| `refs/provincial/` | 省赛迁移笔记；原仓库保持独立，不复制旧赛道 |
| `refs/archive/` | 已被当前方案替代的项目研究记录，仅供历史复核，不参与默认文档检索 |
| `refs/` | 项目参考资料与迁移笔记；非正式运行依赖 |
| `Archive/` | 外来原件及 `technical/` 国赛工作空间技术迭代归档；保留来源，不作为正式运行依赖，文档历史不参与技术归档 |
| `tmp/` | 本机可丢弃的临时产物，不提交；不可作为唯一数据或正式运行依赖 |
| `AIAEC-DevStudio-Open/` | 独立 GitHub 发布仓库；筛选当前源码与文档，不包含本工作空间 Git 历史及本机数据 |
| `国赛规则说明.pdf` | 比赛规则原件，现有路径由文档及实验引用，保留原位 |
<!-- WORKSPACE_LAYOUT:END -->

索引查询和生产回执的职责见[数据集设计](docs/DATASET_DESIGN.md#媒体索引入口)。

### 文件归属规则

创建文件前先判断用途和来源：部署代码、配置与包内测试放对应 `ws/src/aec_<职责>/`；按识别任务组织的训练、推理实验及其测试放 `lab/features/<任务>/`，导航和联调实验分别放 `lab/navigation/`、`lab/integration/`。项目分析与技术决策放 `docs/`，外部资料及其来源说明放 `refs/`；现有 `Archive/` 内原件保留原位，新引入外部资料前记录来源、版本、许可和用途，无法确认时标记待核实。数据和模型分别遵循 [`datasets/README.md`](datasets/README.md) 与 [`models/README.md`](models/README.md)，不把原始数据、权重或凭据提交 Git。

临时结果放 `tmp/` 或任务已有的忽略目录。需要长期使用的结果保存为带来源、范围和验证状态的项目记录。新增顶层条目、数据任务、识别实验分支或 ROS 包时，更新 `config/workspace_layout.yaml` 并运行 `python lab/scripts/contracts/sync_layout_docs.py`。其他子目录按上级目录说明。对象归属、版本策略与生命周期见[工作空间分类](docs/WORKSPACE_ONTOLOGY.md)。迁移文件时同步更新代码、配置、文档和索引中的实际引用。布局检查用 `python lab/scripts/contracts/workspace_layout.py check`，归类检查用 `python lab/scripts/contracts/workspace_ontology.py --check`。

路径自动检查可运行 `python .agents/skills/workspace-file-governance/scripts/check_layout.py`；用 `--path 相对路径` 检查拟使用的路径，或用 `--suggest training-script --task meter` 等按用途建议目录。数据集与 ROS 包角色用 `python lab/scripts/contracts/workspace_layout.py suggest --role <角色> --task <任务>`。目录结构检查另用 `python lab/scripts/contracts/workspace_layout.py check`；它读取 YAML 契约（需要 PyYAML），只读取契约声明的目录名和文件名，在数据树存在时检查数据集结构，不读取文件内容。启用 `.githooks` 的机器默认检查暂存差异格式；仅设置 `AIAEC_FULL_CHECKS=1` 时钩子才检查暂存路径并选择带 PyYAML 的 Python。路径检查仅覆盖可按路径判断的规则，未知职责提示人工核实，不读取被忽略的数据或凭据内容；具体范围及退出码见文件管理 Skill。

V4 数据任务先看[当前生产进度](docs/DATASET_PRODUCTION.md#当前生产进度)，按[从原始采集到导出](docs/DATASET_PRODUCTION.md#从原始采集到导出)运行。各任务目标和组合见[数据集设计](docs/DATASET_DESIGN.md)，数据交付后的入口见[训练流程](docs/TRAINING_WORKFLOW.md#v4-后续窗口的执行方向)。

八项任务的数据规模、回传训练指标与微调得失见[2026-11-19 数据集与模型训练阶段报告](docs/research/DATASET_MODEL_REPORT_20261119.md)，附实际清单与运行回执的聚合摘要。

跨工作空间的设计、推理入口、历史版本、置信度前后对照及 11 月 20 日新增观察见[视觉识别资料与版本对照](docs/CROSS_WORKSPACE_VISION_STATUS_20261120.md)。

## 先做什么

1. 阅读 [任务矩阵](docs/TASK_MATRIX.md)、[小车接入](docs/ROBOT_SETUP.md) 和 [双电脑离线交付](docs/OFFLINE_DEPLOY.md)。
2. 通过 NoMachine 登录实车，按接入文档记录 ROS topic、action、相机、雷达、红外设备和现有工作空间；现有 ROS 包是共享推理骨架，实车适配待接口确认后实现。
3. 在 `lab/features/` 离线验证每类识别；与导航协作方确认两条禁行分支、观察点和终点泊车的接口与现场验证。
4. 在 `ws/src/aec_vision/` 编写实验和 ROS 共用的推理功能，验证后接入一键入口。路线坐标必须现场标定，不沿用省赛坐标。

省赛代码位于相邻的 `catkin_ws` 仓库；其 `ws/src/bag` 的 `move_base`、waypoint、人物、车牌、红绿灯实现可作为迁移输入。国赛规则、赛道和硬件有差异，迁移前需逐项复验。

当前工作重点是全部识别任务，入口见 [识别任务工作区](lab/features/README.md)；训练、独立推理与 ROS 集成的目录划分见 [识别开发设计](docs/PERCEPTION_DEVELOPMENT.md)，采样、后台识别与调度方案见 [异步识别设计](docs/ASYNC_PERCEPTION_DESIGN.md)。

项目的现状、工作边界、阶段方向和 ROS 包命名规范见 [项目方向](docs/PROJECT_DIRECTION.md)。

开源项目与识别方法的调研入口是 [逐任务方法](docs/research/OPEN_SOURCE_METHODS.md) 和 [仪表专项分析](docs/research/METER_VISION.md)。

## 数据标注状态

数据状态按 `source_available`、`annotated`、`candidate`、`auto_validated`、`human_confirmed`、`training_admitted` 分层独立计数；pending 不得因缺少人工接受而自动变成 rejected／excluded。详细定义和写回规则见[标注与准入状态层](docs/DATASET_PRODUCTION.md#标注与准入状态层)，任务入口和批次摘要见[数据区](datasets/README.md#标注状态)。

| 当前范围 | 分层摘要 |
| --- | --- |
| 人物直接审查记录 | source 34 → annotated 23 → candidate 34 条／72 框 → auto_validated 23（16 正、7 负、46 正例框）→ human semantic 3／框或空场 0／掩膜 0 → training admitted 0；pending partial 11 条／26 框、语义 31、标签 34、框／空场 34 条／46 正例框 |
| CarRealPicture 标签 | source 31 图 → annotated 126 条图像—任务记录 → candidate 126 条／94 框 → auto_validated unknown → human confirmed 完整记录 0／框或空场 0／掩膜 0 → training admitted 0；`partial` 2、`unknown` 1 继续 pending |
| SiteCapture 五小样 | source 41 图 → annotated 66 条有标签文件记录 → candidate 82 条／64 框 → auto_validated unknown → human confirmed 0 → training admitted 0；82 条人工复核和 16 条 partial 类记录 pending |

这些数字不是八任务完整标签总数。程序化通过只进入 `auto_validated`；人工语义、框／空场和掩膜分别确认；只有明确失败证据才记 `rejected`／`excluded`。review/manual 重跑必须 merge-preserve，默认校验只读，模板或整份手工记录重建须显式请求、检测非空人工内容并先备份。

## 本机数据集入口

人物数据当前接续点见[当前生产进度](docs/DATASET_PRODUCTION.md#当前生产进度)；各任务的实际入口、计划与执行顺序见[数据集脚本说明](datasets/scripts/README.md#恢复生产入口)。命令从工作空间根目录执行，Python 使用 `.venv/Scripts/python.exe`。本机 Blender 5.2.3 LTS 未加入当前 PowerShell 的 PATH，渲染命令使用以下变量引用它：

```powershell
$DatasetBlender = 'D:\ProgramFiles\Blender5.2\blender.exe'
```

此赋值仅记录软件位置。恢复生产后的 Blender 命令优先使用 OptiX，并设置 `--python-exit-code 1` 使脚本失败返回失败状态。

## 使用与分发

本仓库不包含项目数据、模型和外部资料原件。部分生成与验证命令需自行准备有权使用的素材、权重及运行环境；文档中的本机数据与运行链接在公开克隆中不可用。项目代码许可尚待权利人确定，第三方依赖及资料的许可分别核查。

公开副本的生成统计按随仓库分发的配置重新计算；未分发的本机标注证据保持未知。历史数据与运行结果的聚合观察见阶段报告。运行源码回归测试前先建立临时目录：`New-Item -ItemType Directory -Force tmp | Out-Null`。
