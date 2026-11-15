# 任务训练、版本试验与分发

> 本文提及的部分本地素材、数据和历史运行产物未随本仓库提供；相关名称保留用于说明来源和实验背景。

**数据状态：**当前数据工作目录见[数据区说明](../datasets/README.md)与[数据集设计](DATASET_DESIGN.md#数据集工作目录)。目录结构以 `config/workspace_layout.yaml` 为源，检查用 `python lab/scripts/contracts/workspace_layout.py check`。下文旧批次路径及数量是历史记录，使用前核对实际输入和索引；生产和训练暂停。

开发源位于 `lab/features/<任务>/training/` 与 `lab/training_common/`。清理后 `training_packages/` 保留 `portable_env/`，九个旧任务训练包及旧备份已删除。训练开发读取工作空间代码和数据，运行证据按下文回传。数据来源、候选标注和同源复用范围由[数据区说明](../datasets/README.md)维护；历史数据归档由[技术迭代归档](../Archive/technical/README.md)维护。

## V4 后续窗口的执行方向

2026-11-13，用户确认车牌识别沿用省赛模型。其余八项任务的 train 规模、状态组合和初始来源计划见[八任务训练计划](DATASET_DESIGN.md#八任务训练计划)。验证和测试仅[预留候选父来源](DATASET_DESIGN.md#跨任务共享与预留父来源)，张数待定。V4 按任务设计和制作。此前清理了八个任务的分发包副本，保留 `portable_env/` 及工作区的原始素材、标注、配置和模型。下表记录清理前的候选包，车牌已退出生产范围。完整模型训练主要在分发电脑执行，数据准备、包形成、训练效果及现场验收分别记录。

此前批次记录为八个 V4 包曾落盘、车牌候选尚未落盘。下表是变更前任务包的历史数量；不能把旧包数量直接当作本轮批准的目标或缺口。Blender 列为当时记录的背景图数，不代表新设计下的来源配额已通过。

| 任务包 | 训练 | 开发 | 测试 | Blender 背景 |
| --- | ---: | ---: | ---: | ---: |
| 人物 | 661 | 42 | 32 | 337 |
| 楼名 | 1244 | 50 | 50 | 630 |
| 火焰 | 405 | 60 | 60 | 210 |
| 禁行牌 | 560 | 160 | 80 | 320 |
| 桶开闭 | 273 | 95 | 68 | 175 |
| 垃圾贴纸 | 1310 | 182 | 182 | 1500 |
| 仪表 | 318 | 41 | 41 | 400 |
| 电动车 | 175 | 40 | 40 | 240 |

此前七包已在本机 CPython 3.12／3.14 验证；正式训练和最终测试尚未运行。任务及 split 的实际背景来源构成和原来源按[制作细则](DATASET_PRODUCTION.md#任务分工与阶段边界)检查，验证不等同于模型效果或实车验收。

本轮依据现有规则、资料与逻辑一次性确定数据和训练方案，每个模型只有一次训练机会；重训或微调及初始化仍未确定。车牌沿用省赛模型，不生产 V4 车牌数据集。不安排开发验证、单次模型验证、最终测试或以结果为依据的调参、选模和多次训练；未测模型效果保持未知。训练运行如实际发生，应记录预算、优化器更新和退出状态，零更新不是成功。

**现有入口适用性：** V4 主训练入口 `lab/training_common/engine/runner.py` 在训练选项中固定 `val=True`，默认训练后另跑验证并要求 `best.pt`；`evaluate_after_training: false` 只关闭额外评估。`lab/training_common/engine/data.py` 默认准备 train、val、test，`test_unavailable` 只能省略 test，不能省略 val。分发打包入口 `原根目录工具区的 build_training_packages.py` 已于 2026-11-15 删除，当前没有任务包生成入口。现有 V4 训练入口不适用于本轮无验证路线；不能据此声称单次无验证训练已有可运行入口。V4 最终测试是独立入口，不是训练后自动运行；旧独立入口 `lab/training_common/train.py` 则默认训练后测试，二者不要混同。环境检查或空图前向不证明模型质量。

车牌候选清单 `datasets/exports/plate/v4_master_v4_20261113/manifest.json` 在当前决策前记录训练／开发／测试 1867／393／120 张，其中 1230 张使用 Blender 背景，状态为 `assembled_candidate`；清单及该批导出已清空，不再推进为车牌数据集。旧车牌 OCR 配置及禁行牌分布配置仍引用原导出路径，不能直接读取已删除的输入。

仪表旧候选 `datasets/exports/meter/v41_parametric_station_train_candidates_20261113/` 曾记录 27 张图像和 27 个标签、状态为 `interrupted`；该路径当前缺失。原参数化模型和 `datasets/assets/blender/meter/v4_colored_station_20261113/station.blend` 彩色模型均保留，彩色模型尚待同步来源索引，物理仪表对应关系尚未验证。本轮仪表候选、实拍与混合计划的状态见[制作细则](DATASET_PRODUCTION.md#八任务接入与未受管路径)。

生产状态库中有六条旧回执仍带 `running` 记录：电动车合成、电动车遮罩训练、车牌打包、仪表 V4.1 渲染，以及 2026-11-14 的仪表 `v4_station_thick_close_next20_20261114` 与 `elastic_station_pilot_v2_20261114` 渲染；记录中的 PID 在当前系统均不存在，该状态不表示进程仍在运行。车牌打包回执不属于当前工作范围，不恢复执行；回执作为持久状态保留。其余旧回执恢复前也须核对回执和现有产物。来源数据库在清理任务包后按 `training_packages` 范围更新；2026-11-14 先前一次完整目录重建并同步的 SQLite 快照为 `ready`、628 个分区，其中 508 个来源路径缺失（含 499 个已删除 exports 批次），该快照不能代表本轮清理后的索引状态。在此次清空 `datasets/exports/` 之前，定向重建并同步的 SQLite 快照在 2026-11-14 06:16:38 UTC 为 `ready`、`complete_catalog=true`、648 个分区；该快照不能证明清空后的来源路径仍存在。数据库不会自动监控文件变化；查询及使用前按当前目录和工具状态确认快照范围与新鲜度，需更新时按来源规则刷新受影响分区并同步。数据按任务逐项制作；训练和打包按对应任务的独立授权与实际输入状态开展。

19 项专项与复制路径单元检查使用合成测试输入；引擎另在本机 CPython 3.12 和 3.14 分别完成有界合成夹具训练，均记录到 2 次实际优化器更新。3.12 合成夹具还完成独立验证、带选模依据的冻结和测试，零更新反例标为失败。[3.12 流程记录](../runs/engine_checks/v4_fixture_reason/) · [3.14 训练记录](../runs/engine_checks/v4_fixture_314/) · [零更新反例](../runs/engine_checks/v4_fixture_zero_gate/) 这些检查不证明 V4 真实数据训练或模型效果。

九任务当前配置的初始化权重分别在 CPython 3.12 和 3.14 离线依赖中加载，并各做一次 640 输入的 CPU 空图前向；任务类型、旧包类别顺序及仪表八点形状通过，人物使用通用 COCO 初始化。记录位于 `runs/engine_checks/v4_initialization_inventory_20261112/`；此检查不代表 V4 数据完整、识别效果或 RTX 50 系设备验收。

人物另用 35 张 V4 训练候选和 10 张既有实拍开发区域及其变体，从通用 YOLOv8n 权重直接执行 1 轮 Ultralytics 流程冒烟；本机 RTX 4060、640 输入、batch 4，实际优化器更新 9 次，记录在 `runs/person/v4_actual_data_smoke_20261112/`。该运行未经过 V4 分发入口，未执行最终测试，样本量与训练时长不足以判断模型效果。

V4 暂存训练数据时核对逐图来源、审核证据、任务标签及正负状态，来源缺失或部分真值不进入训练。每次运行快照保留已确认的来源、难度、条件与目标状态；评估按当前集合输出固定阈值分层及组内逐类统计，未记录的条件为未知，多条件组可以重叠。统计按整图分组并计入该图全部目标，混合状态图的结果不能解释为特定单实例状态的召回率。打包与冻结后的来源证据由包内映射和运行快照定位。3.12 和 3.14 的合成分发夹具各完成 1 轮 CPU 训练和验证、3 次实际更新，最终测试均未执行；记录位于 `runs/v4_reviewed_strata_smoke_20261112/`，不代表真实 V4 数据或模型效果。

`by_domain_conditions_class.csv` 将已确认的来源类型、单条件或双条件与类别联合统计，可分别查看偏暗且低饱和条件下的原清晰度实拍、重采样实拍、实拍增强、实景合成和渲染结果；来源或清晰度证据不足时归未知。条件组可重叠，分母和目标仍按整图口径；审核依据会复制进该次运行的数据快照并随冻结方案保留。

**每个模型特制训练。** 各任务按已确认的[训练规模、来源初始计划和状态组合](DATASET_DESIGN.md#八任务训练计划)，确定标签、初始化、增强、优化器、学习率及训练预算；初始化为重训还是微调仍待确定。一次训练机会不以开发验证结果调整预算、早停或选模。既有[逐模型设计表](research/DATASET_COVERAGE_TRAINING.md#逐模型规模与训练设计)是历史候选，不构成本轮对照或验证指令。

八个数据集任务对话分别负责各自的生产，生产完成不自动启动模型训练；共享资源与 Git 按协作协议协调。[阶段边界和生产交接](DATASET_PRODUCTION.md#任务分工与阶段边界)记录本轮一次性方案及证据边界。

| 阶段 | 要完成的工作 | 可以交接的证据 |
| --- | --- | --- |
| 1. 任务设计 | 依据规则、既有错误、来源与标签确定八项数据方案及训练预算；重训或微调未定，车牌沿用省赛模型。 | 记录逻辑、输入、未知和实际约束。 |
| 2. 原件与序列 | 复用已提供的车载序列，按可核对应关系分类；对其余纳入的视频建立完整母序列，细分连续事件片段，混合原视频与车载照片，保留重复簇、缓冲及冻结图约束。 | 成功/失败/未处理区间、来源与片段清单、实际存储量；不以原帧数作为独立样本量。 |
| 3. 逐任务生产 | 按现有资料与逻辑安排正常、困难、负例和未知状态，制作任务专属标签与来源记录。 | 记录实际构成、已知缺陷和未知。 |
| 4. 训练与交接 | 每模型只有一次训练机会；现有 V4 入口不适用本轮无验证路线。 | 记录实际执行状态，不把计划或未测效果写成成功。 |

**分发环境与包体更新。** Windows 训练依赖支持标准 CPython 3.12 和 3.14 x64，启动器和构建器拒绝 3.13；依赖与 RTX 4060 Laptop（本机）、RTX 5070（另一台电脑）的 CUDA 核查见[环境说明](../lab/training_common/portable_env/README.md)。保留包内运行产物和回传证据，目标设备上的整包效果未测。

**包体记录。** 如有交付包，记录任务源码、配置、数据与标签、来源、初始化、环境和输出路径。数据、权重、环境或设备缺失如实注明；不能把能启动等同完整训练或准确率达标。原 V4 构建器要求验证集门控，与本轮路线不符；当前没有任务包生成入口。

四类来源的任务专属采样、Blender 成像校准和真实／合成测试隔离见[接入细则](DATASET_PRODUCTION.md#实拍实景合成与-blender-的接入顺序)。九任务批量候选的数量和来源见[数据集索引](../datasets/README.md#当前素材与来源候选)；来源分层和训练策略尚未完成实证验证。

数据执行依据[数据制作细则](DATASET_PRODUCTION.md)、[当前数据](../datasets/README.md)、本文件及 `config/perception_tasks.json`；[历史概率与训练对照](research/DATASET_COVERAGE_TRAINING.md)仅供既有证据和逻辑参考。优先完成现有素材能支持的工作，不以继续采集作为整项工作的前置条件；缺失真值按任务记录并保留复核队列。每个独立阶段核对并提交 Git，原图、标签导出、权重、环境与 runs 不提交。历史数据沿用已有 ZIP 归档，文档原位维护。

长期目标的包体完成条件是九包逐项具备已核标签、明确来源与 split、任务专用训练策略、正确初始化、可迁移启动和独立输出路径。某包缺少语义复核或关键类别，只能标候选包。实拍精度、训练完成和完整任务验收均依真实结果另记，不因打包完成自动通过。红绿灯 HSV、热异常与各 OCR 保留独立职责，不增设 YOLO 包来凑数量。

## 模型与入口

| 模型 | 任务目录 | 训练脚本 | V3 配置 | V3 最大 epoch |
| --- | --- | --- | --- | ---: |
| 人物两类 | person | `train_person.py` | `person_v3.yaml` | 150 |
| 电动车姿态 | e_bike | `train_e_bike.py` | `e_bike_v3.yaml` | 150 |
| 火焰纸贴 | fire | `train_fire_sticker.py` | `fire_sticker_v3.yaml` | 150 |
| 楼名纸条 | fire | `train_building_name_paper.py` | `building_name_paper_v3.yaml` | 150 |
| 禁行牌具 | sign | `train_sign.py` | `sign_v3.yaml` | 120 |
| 桶开闭 | waste_bin | `train_waste_bin_state.py` | `waste_bin_state_v3.yaml` | 120 |
| 垃圾贴纸 | waste_bin | `train_waste_sticker.py` | `waste_sticker_v3.yaml` | 160 |
| 仪表八点 | meter | `train_meter.py` | `meter_v3.yaml` | 180 |
| 车牌检测 | plate | `train_plate_detection.py` | `plate_detection_v3.yaml` | 120 |

表中记录既有 V3 默认配置；V4 初始化和更新预算依据[数据设计](DATASET_DESIGN.md)及本轮逻辑确定，重训或微调未定。同视频按片段组织和记录复用关系的策略、来源与标签要求见[制作细则](DATASET_PRODUCTION.md#多原件取样与场景混合)；本轮已清理经确认的错误产物，各任务当前候选以实际路径和清单为准，正式数据集尚未完成。

表中脚本位于对应任务的 `training/scripts/`，配置位于 `training/config/`。八个检测模型使用本地 YOLOv8n，仪表使用 YOLOv8n-pose；首轮输入 640、batch 16、seed 42、patience 30。优化器、学习率及增强由配置明确传入，不接受静默忽略的配置键。文字和八点模型关闭镜像增强；默认 workers 0、AMP 关闭以形成可移植的离线基线，设备默认为单张 CUDA 卡。GPU 不可用时报告失败，使用 CPU 须显式修改配置。

仪表八点顺序固定为温度轴心、温度针尖、湿度轴心、湿度针尖、温度最小刻度、温度最大刻度、湿度最小刻度、湿度最大刻度。

车牌现有 `train_cnn_ocr.py` 保持自己的多源 OCR 训练职责。红绿灯采用 HSV/ROI 标定，热异常采用楼层热响应方案，不因统一训练工具而增加 YOLO 模型。楼名与车牌检测不读取文字，仪表 Pose 不直接完成刻度读数，人物／电动车／火焰还需要关联和计数；通用训练指标与完整任务验收分别记录。

火焰既有[误报与过拟合对照方案](research/DATASET_COVERAGE_TRAINING.md#火焰误报与过拟合处理)，记录了红桶与信号灯硬负例及历史训练候选；本轮不执行权重衰减对照、验证或多次训练。

## 旧工作空间运行入口

以下命令说明现有工具行为，不适用于本轮无验证训练路线，不安排运行。

使用已安装匹配依赖的本机环境，或通过[可复制环境](../lab/training_common/portable_env/README.md)的启动器调用。以人物为例：

```powershell
.venv/Scripts/python.exe -B lab/features/person/training/scripts/train_person.py check
.venv/Scripts/python.exe -B lab/features/person/training/scripts/train_person.py train
```

以下是现有入口的历史行为说明，当前 V4 无验证路线不适用这些训练命令。`check` 检查配置、原始 split、类别、标签与本地权重；`train` 保存独立运行记录，以 val 选模并生成验证报告。需要调整配置时使用 `--config <工作空间相对路径>`，恢复中断训练使用 `--resume <本次runs内的training/weights/last.pt>`。恢复时检查与原配置的关系；结束后已剥离优化器的权重不能冒充可恢复检查点。

历史最终测试入口如下；本轮不执行，路径中的运行编号仅作占位：

```powershell
.venv/Scripts/python.exe -B lab/features/person/training/scripts/train_person.py evaluate --split test --weights runs/person/<运行编号>/training/weights/best.pt --run-dir runs/person/<运行编号>
```

`tune` 根据显式配置执行有限试验，仅用 val 选择；`compare --runs <运行目录1> <运行目录2>` 输出已有结果比较。不同样本范围、阈值、输入分辨率、数据来源或软件环境需在比较中保留，不能只报一个提升百分比。测试集不参与调参或选模。

以下为旧调参用法，本轮不运行。调参时另存一份任务配置，在其中加入下列 `tune` 段，然后运行 `tune --config <该配置路径>`。此例最多三次训练、总计两小时；实际 epoch 沿用配置，需要短试验时在该副本中明确降低。总时限覆盖数据准备、训练、评估和诊断；超时保留部分产物并返回 `budget_exhausted`，进程清理可能略超过截止时刻。

```yaml
tune:
  grid:
    lr0: [0.0005, 0.001, 0.002]
  max_trials: 3
  max_hours: 2
```

训练参数搜索默认按检测 Box mAP50–95、仪表 Pose mAP50–95 选取验证候选。对已有权重运行 `tune --weights <best.pt>` 则扫描验证集置信度与 NMS IoU；其配置副本的 `tune` 段使用 `confidence: [0.1, 0.25, 0.4]`、`iou: [0.5, 0.7]`、`max_trials: 6`、`max_hours: 1`、`objective: f1`，与训练参数搜索的配置分开。

## 运行产物

重新构建的各分发包应拥有独立 `runs/`，以脚本所在包为根定位。例如人物为 `training_packages/person/runs/`，仪表为 `training_packages/meter/runs/`。包内模型输出为 `runs/<模型>/<版本与运行编号>/`，启动器日志为本包 `runs/_launcher/`；依赖环境可共享，训练产物按包保存。工作空间源入口使用工作空间的 `runs/<模型>/`；归档试验使用各版本目录内的 `runs/`。

每次模型运行中，`config/` 保存实际生效配置和框架参数，`metadata/` 保存代码版本、环境、设备、执行阶段和成功／失败／中断状态，`logs/` 保存输出和异常，`data/` 保存本次使用的图片、标签、来源对应与数量，`cache/`、`temp/` 接收框架缓存和临时数据。数据原件保持只读；运行目录中的数据副本使原始标签目录免受框架缓存写入影响。

权重和框架训练图表位于 `training/`，TensorBoard 事件位于 `tensorboard/`，独立评估位于 `evaluation/<评估名>/`。保留 loss、Box/Pose 指标、PR/F1 曲线、混淆矩阵、逐类 CSV/图表、逐图预测、预览与失败样本、固定阈值误检漏检和分来源统计；某类别无真值时相应指标为空并注明分母，不能填为成功。仪表关键点指标与框混淆矩阵分别解读。没有 OCR、读数或跨帧真值的评估明确记为未评估完整任务链。每次评估另存图表状态清单；框架未生成的原生曲线或逐类图以独立说明图标出缺失，不重建虚构曲线。历史运行可在 `postprocessing/` 下追加这一清单，保留原始指标和图表。

从其他电脑回传对应包的整个 `runs/`，保留任务、版本和运行编号，禁止覆盖同名运行。运行清单使用相对样本引用；框架生成的绝对路径记录保留其原运行环境意义，重新评估或恢复时由工具重定位运行内的数据路径。恢复训练读取原运行的冻结配置和数据，仅允许修改 batch、workers、device；训练完成后剥离优化器的最终权重不能继续恢复。初始模型和原始训练数据继续由各任务的数据与模型目录管理。

V4 训练前按[制作细则](DATASET_PRODUCTION.md#5-生产记录与交接)完成选材、标注、切分和审查。本轮不执行缩量、等量覆盖和调度训练对照；实际训练如发生，记录优化更新和运行状态。V3 回传范围及固定实拍统计见[实拍诊断](research/FIELD_EVALUATION_V3_20261112.md)。

## V1、V2 与 V3

V1、V2 是本轮实际执行的低预算试验，采用与 V3 相同的完整工程和图表契约。每任务训练使用覆盖类别的同一组 50 张、seed 42、320×320、batch 8、workers 0；V1 为 SGD、lr0 0.01、1 epoch，V2 为 SGD、lr0 0.005、2 epoch。验证和最终测试沿用任务固定集合，不把训练小样本数量误写成评估样本数量。训练预算和学习率同时改变时，比较只说明组合配置效果，不做单因素归因。所有指标来自实际执行，不预设 V1/V2 低分或 V3 高分。

`lab/training_common/smoke_versions.py` 提供 plan、prepare、run、archive。试验准备时冻结当时的引擎与任务配置，每次串行训练并按真实状态记录；失败或中断的版本不能归档成完成版本。完成后放入 `Archive/technical/revisions/<任务>/<模型>_yolo_v1/` 或 v2，其中包含脚本、配置和本地 `runs`；文档不放入技术快照。来源和实际结果在[技术版本索引](../Archive/technical/revisions/README.md)及本文件记录。

V3 是已有完整训练工程及配置；八个完整运行已回传，人物生产训练由用户中止。V4 按本节后续方向独立设计。V1/V2 本机与 V3 分发环境的软件版本分别记录；RTX 50 系电脑采用环境包的 CUDA 12.8 依赖组合；当前分发环境按本机 CPython 3.12 或 3.14 自动选择独立依赖目录。V3 代码准备完成与 V3 模型训练完成是两个状态。

V1/V2 的冻结源码保持运行时原貌；冻结后 V3 补充的路径约束、运行目录迁移、调参总时限及离线代理处理属于工程修改。正式比较 V3 模型时，用同一最终评估环境、数据和阈值重新评估三版权重；现有 V1/V2 原始结果继续保留其环境记录。

## 分发镜像

以下记录既有构建器行为和历史检查。打包入口 `原根目录工具区的 build_training_packages.py` 已于 2026-11-15 删除，当前没有任务包生成入口；本节命令和行为均为历史记录，不能运行。

当时的 V3 刷新命令（工具已删除，不可运行）：

```powershell
# 已删除：原根目录工具区的 build_training_packages.py --refresh-code person e_bike fire_sticker building_name_paper sign waste_bin_state waste_sticker meter plate_detection
```

当时构建器核对工作空间选定源与镜像的文件集合、逐字节图片内容和标签内容，然后刷新受管理的脚本、V3 配置及来源清单；保留已有图片、标签、权重和 `runs`。新增任务包时省略 `--refresh-code`，目标存在则拒绝覆盖。加入 `--include-environment` 将已构建的依赖包复制为同级 `portable_env/`；各任务实现放在包内 `scripts/`，根目录旧入口仅转发。

垃圾贴纸验证／测试采用 24 类共享图案场景；车牌 train/val 组合两批省赛合成来源，test 使用 CCPD 道路照片。这些选择在工作空间 V3 配置中明确列出，镜像不会自行改变数据组合。分发环境的构建、自动解释器发现、ABI/DLL 检查及迁移验证见[Windows 依赖包说明](../lab/training_common/portable_env/README.md)。

V4 使用构建器的 `--v4-manifest` 指定单任务已审来源和初始化；构建器先暂存并核对数据，复制到包含空格的新路径后，在 3.12、3.14 离线环境实际加载权重、做空图前向及检查入口，并用实际启动器核查自动发现默认兼容解释器，通过后才替换，旧包和其 runs 保留备份。此前候选包的重建检查只覆盖当次快照；九个旧任务包已撤回，按新背景要求重做。本轮不安排测试或验证选模；正式训练及模型效果的实际状态分别记录。旧 `--refresh-code` 保持 V3 用途，不得把其输出称为 V4。合成夹具的迁移与自动发现检查见 `runs/engine_checks/v4_relocated_fixture_20261112/result_final.json`；其中启动器在故意设置的无效时限处停止，未执行训练，3.14 自动发现未单独隔离验证。

## 工程验证

2026-11-11，九个分发包通过新便携环境的 `check`，逐包记录原位于各自 `runs/_verification/preflight.json`，随已删除的任务包移除；合计 41,277 张图片与标签完成结构检查，镜像刷新前另逐字节核对来源图片和标签。训练工具 46 项检查通过，涵盖标签与类别契约、输出路径、数据来源映射、镜像刷新、启动失败清理、运行迁移和调参超时。便携环境另有四项兼容与隔离检查，以及换目录后的原生库、CUDA 算子、DataLoader 和 TensorBoard 检查，证据见其说明。

人物分发包通过新依赖环境执行了 CPU 8 张训练／8 张验证、320 输入、1 epoch 的集成检查，保存训练图表、TensorBoard、权重与验证结果，原目录为 `training_packages/person/runs/_verification/v3_environment_check_20261111_103746_284299/`，已随人物包移除。最终引擎另在同一运行内完成 `evaluation/val_final_engine/` 评估与图表清单检查。这是当时的启动与环境检查；后续 V3 生产状态见[回传记录](research/FIELD_EVALUATION_V3_20261112.md#v3-训练行为)。

中断与迁移续训使用本机旧训练环境在 CPU 上实际验证：8 张训练、4 张验证、计划 2 epoch；第 1 epoch 保存后主动中断，将完整运行复制到新位置，再用运行内数据完成第 2 epoch。原始 checkpoint 和指标文件内容保持不变，新运行 CSV 的 epoch 为 `[1, 2]`，证据见 `runs/engine_checks/resume_migration_20261111_104833/summary.json`。

## 2026-11-11 冒烟试验结果

九个模型的 V1/V2 共 18 次训练及完整 val/test 评估已完成并归档。每对版本使用相同的 50 张训练图，训练子集保留全部类别；垃圾贴纸覆盖 24 类。执行环境为 Windows、RTX 4060 Laptop、Torch 2.2.2+cu121、Ultralytics 8.2.103。训练加评估进程累计约 30.23 分钟。

下表为各任务固定测试集上的 Box mAP，取值 0–1；不是分类准确率或完整任务成功率。点击任务查看该任务的 val/test 比较数据和图表。

| 模型 | test 图片 | V1 mAP50 | V2 mAP50 | V1 mAP50–95 | V2 mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [人物](../lab/features/person/training/runs/person/v1_v2_comparison_20261111/comparison.json) | 200 | 0.006143 | 0.009515 | 0.003756 | 0.005983 |
| [电动车](../lab/features/e_bike/training/runs/e_bike/v1_v2_comparison_20261111/comparison.json) | 200 | 0.010233 | 0.011764 | 0.006172 | 0.007429 |
| [火焰纸贴](../lab/features/fire/training/runs/fire_sticker/v1_v2_comparison_20261111/comparison.json) | 120 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [楼名纸条](../lab/features/fire/training/runs/building_name_paper/v1_v2_comparison_20261111/comparison.json) | 120 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [禁行牌](../lab/features/sign/training/runs/sign/v1_v2_comparison_20261111/comparison.json) | 200 | 0.013777 | 0.018612 | 0.008574 | 0.012602 |
| [桶开闭](../lab/features/waste_bin/training/runs/waste_bin_state/v1_v2_comparison_20261111/comparison.json) | 120 | 0.049426 | 0.054491 | 0.036494 | 0.046427 |
| [垃圾贴纸](../lab/features/waste_bin/training/runs/waste_sticker/v1_v2_comparison_20261111/comparison.json) | 120 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [仪表](../lab/features/meter/training/runs/meter/v1_v2_comparison_20261111/comparison.json) | 500 | 0.001824 | 0.064969 | 0.000512 | 0.046380 |
| [车牌检测](../lab/features/plate/training/runs/plate_detection/v1_v2_comparison_20261111/comparison.json) | 2800 | 0.000560 | 0.001065 | 0.000238 | 0.000500 |

仪表 Pose mAP50 为 V1 0.302557、V2 0.272389；Pose mAP50–95 为 V1 0.054447、V2 0.044231。本次 V2 关键点指标下降。两版在诊断置信度 0.25 下没有匹配检测框，关键点误差缺少可计算的配对，记录为 null。

九组比较分别核对了 val/test 样本来源、类别顺序与指标库版本。原始汇总为 `runs/smoke_versions/training_iterations_20261111/test_results.json`，运行与归档对应见同目录 `suite.json`，完整性检查见 `artifact_audit.json`。训练数据与测试数据的来源关系沿用[数据区说明](../datasets/README.md)；该表只描述这些固定集合。后续八个 V3 完整运行已经回传，人物生产训练中途停止，见[回传记录](research/FIELD_EVALUATION_V3_20261112.md#v3-训练行为)。
