# 开发约定

本工作空间目录为 `D:\DevForge\Programming\Languages\Python\VSCode\AIAEC-DevStudio`。2026-11-15 前的会话、日志和历史记录写作 `D:\DevForge\Programming\Languages\VScode\AIAEC-DevStudio`，该旧路径已不存在，指的就是当前目录。

- 正式运行目标为 Ubuntu 20.04 + ROS Noetic + Python 3；Windows 上仅做离线检查，不声称实车已验证。
- `ws/src/` 只放可部署 catkin 包。`lab/`、`refs/`、`datasets/` 不应成为正式运行时依赖。
- 导航使用 ROS `move_base` 和现场标定的地图、坐标；禁止把省赛坐标直接作为国赛路线。
- 禁行牌决定路线分支，未可靠识别时应停车并记录故障，不能盲走受限路径。
- 红灯停在等待线前；碰撞、压线、超时均是比赛风险。速度、重试、超时要有明确上限。
- 识别结果、图片标注和终端播报要与国赛规则逐项对应；失败不应伪装为零或成功。
- 新增相机或红外设备前核对传感器数量限制，记录型号、话题、分辨率与标定结果。
- 不提交原始数据、模型权重、密钥、虚拟环境及 catkin 生成目录。
- V4 使用的原始图片、视频及数字素材以 `datasets/raw/` 为来源，允许使用 `Archive/Assets/`；Blender 场景和模型允许来自 `datasets/assets/blender/`。标注、清单、数据库、代码及由允许来源派生的中间结果是不同职责的输入和记录，保留其实际位置和父来源，不因不在上述原始来源目录而认定为原始素材或自动删除。其他历史或当前原始来源按证据报告。
- 识别训练以合成数据为主，同时纳入直接实拍，实拍应占相当比例；缺少标签的当前实拍按任务重建标注，本轮已有标签按实际状态使用，未知不伪造；合成包括用实拍或数字图片进行图像处理合成、Blender 渲染，以及渲染图与实拍的混合。各任务依据已有来源、标签状态和覆盖需求确定具体组成，不设统一数字。数据记录同时注明来源谱系和生成方式：直接相机采集、实拍图像增强、基于实拍拟合模型的渲染、实拍场景合成等分别标记；实拍来源的派生数据保留来源关系并单独报告，不等同于新的独立相机采集。同一原始照片、视频及其派生样本可按用户确定的训练方案用于不同集合或任务；记录已知复用范围，同源验证结果不冒充独立采集或独立场次评估。
- V4 数据设计逻辑至上：先由任务目标确定应学习的特征及不应依赖的条件，直接设计合理、保持目标语义和标签正确的状态及联合组合；不要求为每个人为构造的组合逐一找到实拍对应证据。再查询数据库选择素材并按实际入口执行，来源、生成参数、标签、已知缺陷及产出如实记录；设计权重和假设不冒称实测分布。贝叶斯只为已决定的可行组合分配预算，不决定任务目标或凭空证明效果。每个模型只有一次训练机会，重训还是微调尚未确定。不安排常规开发验证、单次模型验证或最终测试，不以模型验证结果安排重训或选模；用户明确授权的有限素材、小样或代码契约测试按指定范围执行；未测效果如实记为未知。安全门控继续执行，训练入口与该约束不符时如实报告，不能假称已适配。
- 数据和审查按 `source/available`、`annotated`、`candidate`、`auto_validated`、`human_confirmed`、`training_admitted` 分层独立计数，并同时报告 `pending`。缺少人工确认、未复核或未准入只能保持 pending，不得自动改写成 rejected/excluded，也不得清空已有候选或标注。程序化/自动验证不得填充人工层；人工语义确认与实际框/掩膜确认分开记录。`rejected/excluded` 只用于有实际失败证据的记录。
- V4 数据集当前纳入八个需设计数据集的任务；车牌识别采用省赛模型，不生产车牌数据集。各任务按关键状态、可用父来源和分组切分确定规模与来源组成；V4 打包记录 Blender 背景的实际数量与来源，不执行统一占比门槛。讨论记录见 `docs/DATASET_DESIGN.md`。
- 训练目标与组合轴分开记录：类别、关键点、OCR、计数等模型要学习的对象放 `config/task_targets.yaml`；光照、视角、遮挡、背景、截断等用于排除或平衡无关噪声的条件放 `config/combination_axes.yaml`。同一词在不同任务中角色可不同（如电动车直立/倒伏是电动车目标类别、人物任务的遮挡干扰轴）；覆盖与配额缺口用 `python lab/scripts/contracts/state_coverage.py` 查看，不把组合轴当类别或把目标当噪声。
- 全量 train 计划设计草案在 `config/full_train_allocation.yaml`，生成文档为 `docs/FULL_TRAIN_PLAN_DESIGN.md`；它是 PlanDraft，不覆盖也不被生产守卫当作 ProductionPlan。编译成 `production_plan.json` 前核对来源容量、父组 80% 上限和首批执行顺序。
- 对象分类学以 `config/workspace_ontology.yaml` 为唯一源：类层次、所有者、权威源、版本策略、可变性、生命周期和关系在此定义；检查用 `python lab/scripts/contracts/workspace_ontology.py --check`，生成文档 `docs/WORKSPACE_ONTOLOGY.md`。新增对象先选父类和归属，不另建同义目录或版本命名目录。
- 每个任务的父子源与 split 预留以 `datasets/scripts/<任务>/config/source.yaml` 的 `pairing` 为权威记录；计划编译前运行 `python lab/scripts/contracts/source_pairing.py --check`，`plan_alignment.status=stale` 或 split 冲突保留为待决项，不能只改文档假装闭环；`pairing` 不是已执行计划。配对校核以 `lab/scripts/contracts/source_pairing.py` 为单一 checker；`source_ledger` 只作数据访问/实验，不作为第二套校验器。
- 顶层目录、数据集任务、识别实验分支和 ROS 包结构以 `config/workspace_layout.yaml` 为唯一编辑源：新增或调整这些目录先更新契约，运行 `python lab/scripts/contracts/sync_layout_docs.py` 生成 README 目录表和数据集目录区块，再用 `python lab/scripts/contracts/workspace_layout.py check` 核对。其他子目录按各自 owner 文档；生成区块不手工维护；检查只读目录名和声明文件，不读取数据内容。
- `datasets/blender_source_registry.sqlite3` 是 Blender 源资产分类、生命周期、已核实关系和明确标记的 manifest 声明/未核 claim 的权威记录；使用 `lab/scripts/blender/blender_source_registry.py` 查询或登记，不另建同义台账。新增、更正、归档和路径迁移应保留旧路径及证据引用，关系记录引用原始清单路径和 `record_index`（有此字段时）；历史 manifest 只证明生成时声明，不覆盖当前资产状态。`query_cache.sqlite3` 仍是可重建媒体观察查询库，`production_state.sqlite3` 保存生产回执，两者均不覆盖 Blender 登记库。数据库不监控文件系统，查询时按需核验路径和索引新鲜度。其他来源域在各自权威范围确定前，不将 Blender 登记库外推为全工作空间来源主库。生产入口的代码门控覆盖情况和具体入口见 `docs/DATASET_DESIGN.md`。
- 数据生成和打包入口使用 `lab/training_common/production_guard.py` 查询已就绪的来源数据库，按输入、参数、代码和输出完成回执复用结果；各任务接入与验证按实际状态记录。当前生产守卫没有全局原始来源根目录白名单，具体入口的额外约束按实际实现记录，不能据此声称八任务已统一门控。生产守卫首次操作时创建 `datasets/annotated/media_index/production_state.sqlite3` 保存回执；该库是持久状态，不得作为可重建查询缓存清理。生成入口不自动全扫；索引变化按前述规则显式刷新。训练包只保留训练、验证、测试和可视化入口，数据生成通过工作空间入口执行。
- 版本管理与历史依据使用 Git；国赛工作空间内有明确迭代的数据集、脚本、模型可按 `workspace-history-archive` 复原和归档，保留原路径与来源对应关系。禁止将相邻工程独立产生的历史数据和模型放入本工作空间的归档；文档不参与历史归档；不引入散列校验或其他恢复机制。
- 工作空间文档、需求、注释、日志、输出、索引、报告和回复均直接记述事实与要求：缺失写缺失，中途停止写中途停止，截断写截断。涉及事实记述时使用文档治理 Skill；纠正误记时原位删除或改写，并同步受影响内容。修订历史交给 Git。
- 除根目录等头部文档外，文档及脚本不引用工作空间外的本地文件路径。脚本绝不加载或执行工作空间外的脚本文件；确需使用的源码先核对来源和许可，复制到工作空间合适位置并独立维护修改。系统解释器、已安装软件及库不属于外部脚本文件；数据、模型等仍须遵循项目的数据与部署约定。
- 每完成一个独立、可验证的工作阶段，检查本阶段变更并立即自动执行 Git 提交，无需另行询问；提交信息准确说明该阶段内容，不混入无关改动。
- 现已具备真实数据采集条件。用户已指定使用现有素材与合成生产时，先查询数据库复用已有记录，利用可控合成解决覆盖缺口；只有现有素材和合成都无法解决且实际阻塞任务的具体事实，才向用户提出采集任务。确需采集时说明拍摄对象、场景布置、视角、光照、数量或时长、真值及保存位置等必要信息；能够自行取得的数据及时采集和整理。不要仅以“还需现场验证”结束，或把真实数据、实验及验证作为拒绝推进整个任务的理由；并行完成不依赖缺失输入的设计、实现和检查，取得数据后按用户授权继续相关工作；本轮不以常规数据或模型验证选模或迭代；用户明确授权的有限测试按指定范围执行。真实数据和验证结果必须如实记录，不能虚构采集、测量或验收结果；涉及行车和安全门控的既有约束继续执行。
- 工作空间文档的查询、新增、修改、删除、归档，以及需求变更引起的文档更新，统一使用 `workspace-doc-governance` skill（`.agents/skills/workspace-doc-governance/SKILL.md`）；需求变化时同步处理受影响文档，遵循其权威来源与生成规则。纯查询保持只读；无法处理或证据不足时按 skill 明确说明原因、影响和所需信息。
- 需要归档、复原、对照或整理国赛工作空间的技术迭代及素材时，自动使用 `workspace-history-archive`（`.agents/skills/workspace-history-archive/SKILL.md`），记录来源、边界、差异和缺失证据。
- 涉及归档、素材分类及错放检查时，在子代理可用时委派项目角色 `archive_curator`，按任务授权与现有 Skill 核实和处理；该角色不常驻后台，Git 写操作由主代理串行协调。
- 开始实验、编写脚本或其他可能创建、移动、引入、整理文件的任务时，无需用户另行提醒，动手前使用 `workspace-file-governance` skill（`.agents/skills/workspace-file-governance/SKILL.md`），按 `README.md` 目录表与 `config/workspace_layout.yaml` 确定归属、来源、用途与是否可提交；数据集任务先运行 `python lab/scripts/contracts/workspace_layout.py check --task <任务>`，标注根和配置要求看 `docs/DATASET_DESIGN.md#数据集工作目录` 的生成区块。优先使用现有目录，不能确定时报告原因并避免任意放置。移动前检查引用、生成器和运行路径，完成后同步修复并核对 Git 变更；不自动移动或删除用户数据、凭据、外来原件及未识别文件。新增目录须有明确职责并更新契约或所属目录说明。
- 工作空间级维护脚本放在 `lab/scripts/` 的 `contracts/`、`media/`、`blender/`、`data/`、`audit/`、`common/` 分类中，不新增根目录 `tools/`；新增脚本先按职责选择分类，目录结构以 `config/workspace_layout.yaml` 为源，分类规则见 `lab/scripts/README.md`。
- 识别任务需求以 `config/perception_tasks.json` 为唯一编辑源；修改需求或共享推理模块后运行 `python lab/scripts/contracts/sync_docs.py`，生成的任务文档不手工维护。
- 编号、任务口径、计数单位与标注状态层以 `config/id_policy.yaml` 为唯一编辑源，生成 `docs/ID_AND_COUNT_POLICY.md`；检查用 `python lab/scripts/contracts/id_policy.py --check`。新需求项使用 `TASK-RNN`，语义 ID 与裸数字 ID 只作冻结别名；识别任务、数据任务、模型包三套口径、图片/实例/父组/状态/预算/容量/配额单位及状态层语义在此定义，其他文档不另立编号族或第二套状态口径。
- 标注事实按 `source_available`、`annotated`、`candidate`、`auto_validated`、`human_confirmed`、`training_admitted` 六个相互独立的层计数；每张摘要必须在同一计数单位下同时展示六层、`pending` 数及适用的人工语义/框/掩膜确认计数。各层可以重叠，不得相加冒充总量。
- `pending` 表示未完成或未确认；缺少 `accepted_by`、尚未人工复核、尚未准入或当前无法判定时只能记 `pending`/`unknown`，不得自动改写既有标注或候选记录为 `rejected`/`excluded`。自动或程序化验证只填 `auto_validated`，不能填 `human_confirmed`；人工语义确认与人工框、掩膜确认分别记录。`rejected`/`excluded` 只有在存在明确实际失败证据时才可使用，人工最终接受为 0 不得写成标注或标签为 0。
- 共享推理核心直接编写在 `ws/src/aec_vision/src/aec_vision/`，独立实验及 ROS 适配层导入同一模块。
- 新增功能、编写或修改脚本逻辑、改变共享模块/接口行为前，自动使用 `shared-code-interfaces`（`.agents/skills/shared-code-interfaces/SKILL.md`），从所属职责区域搜索现有能力和调用方，再选择直接复用、局部扩展或必要的新实现；不为复用率建立无实际需求的通用框架，不将不同语义强行抽象。
- 涉及数据素材的采集、导入、标注、增强、切分、训练、评估、指标比较及相关数据/指标逻辑变化时，自动使用 `workspace-experiment-governance`（`.agents/skills/workspace-experiment-governance/SKILL.md`），复用既有实验工具并检查来源、泄漏、覆盖范围和结论证据；纯查询保持只读。
- 影响正式运行行为的共享推理、模型加载、部署配置/依赖、包结构或 ROS 接口变化时，自动使用 `workspace-offline-delivery`（`.agents/skills/workspace-offline-delivery/SKILL.md`）做交付影响评估；交付、部署或验收任务再执行适用环境的验证，不因 Skill 自动触发车辆运动或完整训练。
- 需要连接正在运行的 Blender 执行 bpy 代码、渲染导出或核对 Blender 资产时，使用 `blender-mcp-dsh`（`.agents/skills/blender-mcp-dsh/SKILL.md`）：它只适用 DSH 的 profile patch 与 `lab/scripts/blender/blender_mcp_bridge.py` 桥接，不适用 Codex/Claude 的 blender-mcp 配置；正式数据生成仍走 `datasets/scripts/<task>/` 入口与生产守卫，单帧预览不冒充数据集批次。
- 任务开始时先按下表判断实际影响，再组合适用 Skill；同一阶段的路径检查、来源检查、生成文档同步、报告及提交统一执行一次，相关证据变化后才重做。互相引用不要求重复进入流程，无变更不创建空提交。无法处理时明确未完成项、原因、影响与所需信息，不重复建立台账或文档副本。
- 项目自建 ROS 包统一采用 `aec_<职责>` 命名，使用小写英文字母和下划线，职责名称简短明确；包目录、`package.xml`、CMake 项目名和同包 Python 导入名保持一致。命名与项目方向见 `docs/PROJECT_DIRECTION.md`。

## 任务开始时的 Skill 分流

用户要求跨聊天协调、读取进展、交接或组织窗口内子代理时，使用 `workspace-collaboration`（`.agents/skills/workspace-collaboration/SKILL.md`）。本工作空间各对话窗口可按需启用子代理处理局部问题，明确每个子代理的职责和独占文件范围，由所属窗口负责人汇总并核对成果；各窗口及其子代理共享任务边界，不覆盖其他工作，Git 暂存、提交和切分支由协调方串行处理。这项子代理授权及调用 Skill 本身不授权创建聊天、跨聊天发送消息或迁移执行环境；这些操作按用户直接授权和当前工具约束执行。

| 场景 | 自动选用与边界 |
| --- | --- |
| 只查文档/比较已有实验指标 | 文档治理；指标比较加实验治理的证据检查，保持只读，不训练或新建报告 |
| 归档、复原或对照国赛技术迭代 | 历史归档；涉及文件整理、文档或数据时分别结合文件、文档或实验治理 |
| 新写/修改普通维护脚本 | 文件治理 + 共享代码维护，从所属工具区查复用；不自动进入训练或交付 |
| 新建/调整目录、数据集生产 | 文件治理 + 目录契约检查（`python lab/scripts/contracts/workspace_layout.py check/suggest`）；改契约后运行 `python lab/scripts/contracts/sync_layout_docs.py` |
| 调整训练目标、组合轴或配额 | 实验治理；目标与轴分文件，覆盖/缺口用 `python lab/scripts/contracts/state_coverage.py`，不把缺口直接当配额 |
| 运行已有实验脚本 | 文件治理处理产物位置，实验治理处理数据与运行；不改代码则不重新设计接口 |
| 连接运行中的 Blender 渲染/查场景/跑 bpy | `blender-mcp-dsh` 先做连接三步核对（socket、execute、profile patch 注册）；产物落点按文件治理，涉及数据素材加实验治理；单帧预览不冒充数据集批次 |
| 编写实验脚本或训练专用结构 | 文件治理 + 共享代码维护；涉及数据/指标加实验治理，仅研究用途不自动进入部署 |
| 改共享行为/返回字段/正式配置或模型 | 共享代码维护检查契约，交付 Skill 评估影响；涉及实验加实验治理，涉及文档加文档治理 |
| 只打包交付 / 实际部署验收 | 交付 Skill 按本次范围执行；打包不自动安装或上车，无目标机不阻塞单纯打包完成 |
| 纯注释/格式修正 | 确认无行为或契约影响后，不展开无关测试与验收；共享模块修改仍遵守既有文档同步要求 |

文件治理负责落点，代码 Skill 负责复用与契约，实验 Skill 负责数据与指标，交付 Skill 负责环境与验收，文档 Skill 负责权威记录，`blender-mcp-dsh` 负责 Blender 连接与协议。按此分工共享检查结果；后续发现影响扩大再追加相应 Skill。注释若揭示真实契约变化，不能按纯格式修改处理。
