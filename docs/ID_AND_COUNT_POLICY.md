# 编号与计数口径

> 状态：`active`。本文由 [`config/id_policy.yaml`](../config/id_policy.yaml) 生成，修改请改 YAML 后运行 `python lab/scripts/contracts/id_policy.py --write`；检查用 `python lab/scripts/contracts/id_policy.py --check`。
>
> 本文定义编号命名空间、三套任务口径、计数单位、标注状态层和权威源，不修改需求、计划、数据或产量。

## 需求 ID

新需求项只使用规范族 `TASK-RNN`，模式 `^[A-Z][A-Z0-9_]*-R[0-9]{2,3}$`。语义 ID 和裸数字 ID 是冻结别名，保留原含义，不再新增；修改需求文字时保留原 ID。

当前需求配置共 **72** 条：规范 `TASK-RNN` **35** 条，冻结语义别名 **22** 条，冻结裸数字别名 **15** 条。

| 任务 | 规范 `TASK-RNN` | 冻结语义别名 | 冻结裸数字别名 |
| --- | --- | --- | --- |
| `traffic_light` | `TRAFFIC_LIGHT-R01`、`TRAFFIC_LIGHT-R02`、`TRAFFIC_LIGHT-R03`、`TRAFFIC_LIGHT-R04` | `TRAFFIC_LIGHT-METHOD` | `TRAFFIC_LIGHT-1` |
| `person` | `PERSON-R01`、`PERSON-R02`、`PERSON-R03`、`PERSON-R04` | `PERSON-COUNT`、`PERSON-TARGET-SCOPE`、`PERSON-MIN-SCALE`、`PERSON-DATASET-READY`、`PERSON-TRAINING-V3` | `PERSON-1`、`PERSON-2` |
| `sign` | `SIGN-R01`、`SIGN-R02`、`SIGN-R03` | `SIGN-DATASET-READY`、`SIGN-TRAINING-V3` | `SIGN-1` |
| `waste_bin` | `WASTE_BIN-R01`、`WASTE_BIN-R02`、`WASTE_BIN-R03`、`WASTE_BIN-R04` | `WASTE_BIN-SCOPE`、`WASTE_BIN-DATASET-READY`、`WASTE_BIN-TRAINING-V3` | `WASTE_BIN-1`、`WASTE_BIN-2`、`WASTE_BIN-3` |
| `fire` | `FIRE-R01`、`FIRE-R02`、`FIRE-R03`、`FIRE-R04`、`FIRE-R05` | `FIRE-DATASET-READY`、`FIRE-TRAINING-V3` | `FIRE-1`、`FIRE-2` |
| `thermal` | `THERMAL-R01`、`THERMAL-R02`、`THERMAL-R03`、`THERMAL-R04` | `THERMAL-FACT`、`THERMAL-THRESHOLD`、`THERMAL-OBSERVE` | — |
| `meter` | `METER-R01`、`METER-R02`、`METER-R03`、`METER-R04` | `METER-DATASET-READY`、`METER-TRAINING-V3` | `METER-1` |
| `plate` | `PLATE-R01`、`PLATE-R02`、`PLATE-R03`、`PLATE-R04` | `PLATE-DATASET-SCOPE`、`PLATE-TRAINING-V3` | `PLATE-1`、`PLATE-2`、`PLATE-3` |
| `e_bike` | `E_BIKE-R01`、`E_BIKE-R02`、`E_BIKE-R03` | `E_BIKE-DATASET-READY`、`E_BIKE-TRAINING-V3` | `E_BIKE-1`、`E_BIKE-2` |

## 配对编号

| 命名空间 | 状态 | 用途 |
| --- | --- | --- |
| `PAIR-RNN` | 规范 | 新增配对设计需求 |
| `PAIR-VNN` | 规范 | 新增配对校核规则；`source_pairing.py` 的新错误码使用此族 |
| `R1-R12` | 冻结 | 现有配对设计需求 |
| `V1-V12` | 冻结 | 现有配对校核规则；V11 尚无自动 checker，编译前专项核对 |
| `BSL001-BSL030` | 冻结 | `source_ledger` 派生报告诊断；不是准入 gate |
| `BSL010`、`BSL015`、`BSL024`、`BSL027`、`BSL028`、`BSL029` | 保留空号 | 当前实现未使用；不得复用为其他含义，待 ledger 维护者登记 |

配对口径说明：

- 新增配对设计需求使用 PAIR-RNN，新增校核规则使用 PAIR-VNN。
- 现有 R1-R12、V1-V12 和 BSL 编号冻结；不再新增裸 R、V 或 BSL 编号。
- source_pairing.py 是配对单一 checker；source_ledger 的 BSL 只作派生报告诊断，不是第二套 gate。
- V11 目前没有自动 checker，编译逐 split 计划前仍需专项核对。

## 三套任务口径

| 口径 | 成员 | 来源 |
| --- | --- | --- |
| 识别任务 | `traffic_light`、`person`、`sign`、`waste_bin`、`fire`、`thermal`、`meter`、`plate`、`e_bike` | `config/perception_tasks.json` |
| 数据任务 | `person`、`e_bike`、`fire`、`building_name_paper`、`sign`、`waste_bin_state`、`waste_sticker`、`meter` | `config/workspace_layout.yaml` |
| 模型包 | `person`、`e_bike`、`fire`、`building_name_paper`、`sign`、`waste_bin_state`、`waste_sticker`、`meter`、`plate` | `config/task_targets.yaml` + `model_packages_extra` |

识别任务到数据任务的映射：

| 识别任务 | 数据任务 |
| --- | --- |
| `traffic_light` | — |
| `person` | `person` |
| `sign` | `sign` |
| `waste_bin` | `waste_bin_state`、`waste_sticker` |
| `fire` | `fire`、`building_name_paper` |
| `thermal` | — |
| `meter` | `meter` |
| `plate` | — |
| `e_bike` | `e_bike` |

任务口径说明：

- 识别任务是 config/perception_tasks.json 的九项，包含红绿灯、热异常和车牌，不把楼名纸条、垃圾贴纸拆成独立识别任务。
- 数据任务是 V4 需要自建数据的八项；fire 拆为火焰贴纸与楼名纸条，waste_bin 拆为桶开闭与垃圾贴纸。
- 模型包是八个数据任务模型加车牌省赛模型；红绿灯使用 HSV/ROI，热异常不设 YOLO 包，均不计入九个模型包。

## 计数单位

| 单位 | 定义 | 权威源 |
| --- | --- | --- |
| `image` | 一张最终训练图片；同一图片被多个任务复用时，各任务分别计图片，但父来源不因此成为新的独立采集。 | datasets/scripts/<任务>/config/dataset.yaml 与 config/full_train_allocation.yaml |
| `instance` | 图片中的一个独立目标实例；同一图片可以有多个实例，不能与图片数互相替代。 | config/task_targets.yaml 与任务标签/清单 |
| `parent_group` | 一个独立采集场次或可独立归组的父来源；同一视频的帧、裁剪、增强和渲染不增加父组。 | datasets/scripts/<任务>/config/source.yaml 的 pairing |
| `state` | 一个可编辑场景、渲染状态或组合状态；不等于最终图片张数，也不等于父组数。 | 任务 render_plan.json、production_plan.json 或 source.yaml |
| `plan_budget` | 计划声明的目标产量或预算；可能尚未生产，不能当作实际产量。 | datasets/scripts/<任务>/config/*plan*.json 的 budget |
| `plan_capacity` | 现有计划 strata 的 max_count 之和；表示当前计划结构最多能承载的图片数，不是父源容量。 | datasets/scripts/<任务>/config/*.json |
| `quota` | 设计分配给某任务或某组的图片数，来自 PlanDraft 或 ProductionPlan；不是实测分布或模型效果。 | config/full_train_allocation.yaml 或 production_plan.json |

## 标注状态层

六层是相互独立的计数视图，同一记录可以同时属于多层；不得把它们相加当总量，也不得用一层计数替代另一层。

每张摘要必须声明计数单位，六层和 pending 只能在同一单位、同一任务范围与同一时间口径下并列展示。

| 层 | 定义 | 权威源 |
| --- | --- | --- |
| `source_available` | 已纳入指定来源范围、文件可访问且来源关系可追溯的图片或实例；不要求已有标注。 | 任务 source.yaml、原始清单和实际文件核验 |
| `annotated` | 已存在明确标签载荷的图片或实例；标签文件、JSONL 记录、polygon 或 mask 需能核对到当前图片，单纯清单声明不单独提升本层。 | 任务标签、JSONL、polygon 与掩膜文件 |
| `candidate` | 由模型、图像处理或程序生成的候选框、候选实例或候选审查记录；候选不等于人工确认。 | 任务候选清单、review selection 与候选标注记录 |
| `auto_validated` | 已通过明确程序化或自动校核的图片、实例或记录；只填自动验证层，不填人工确认层。 | 自动检查指标、reason code 与代码版本 |
| `human_confirmed` | 当前记录所需人工语义确认以及适用的人工框或掩膜确认均已完成并留下确认人证据的图片或实例。 | 人工复核记录、确认人、确认时间和逐项确认结果 |
| `training_admitted` | 按当前训练准入规则明确接纳的图片或实例；不能仅由文件存在、候选生成、自动通过或人工确认中的单项事实推定。 | 训练准入清单、当前 ProductionPlan 与任务实际消费入口 |

`pending` 不是第七个成果层，而是六层计数旁必须同时展示的未完成或未确认数量。

尚未完成当前层所需确认或证据、但也没有实际失败证据的状态；pending 不是 rejected 或 excluded。

缺少 accepted_by、尚未人工复核、尚未准入或当前工具不能判定时记 pending；不得自动改写既有标注或候选记录。

### 人工确认维度

| 维度 | 定义 | 规则 |
| --- | --- | --- |
| `semantic` | 确认对象类别、存在性、任务归属和语义是否正确的独立人工维度。 | 程序化接受只支持 auto_validated，不能填充本维度。 |
| `box` | 确认目标框的位置、范围和数量是否正确的独立人工维度。 | 语义确认不得替代框确认；框仍待确认时保留 pending 框状态。 |
| `mask` | 确认掩膜、多边形或实例覆盖是否正确的独立人工维度；不要求每个任务都存在掩膜。 | 语义或框确认不得替代掩膜确认；任务适用掩膜但尚未确认时保留 pending 掩膜状态。 |

人工语义、人工框和人工掩膜分别计数；human_confirmed 只有在当前记录所需维度全部完成时才成立。

### 负状态证据门槛

允许的负状态：`rejected`、`excluded`。只有存在实际失败证据时才可记 rejected 或 excluded；证据应能指向失败的检查、标签、来源、规则或人工判断。

缺少确认、未审核、未准入、空结果或无法判定都只能是 pending 或 unknown，不得因未完成而自动转为负状态。

### 摘要契约

所有摘要必须同时列出计数单位、六层计数（`source_available`、`annotated`、`candidate`、`auto_validated`、`human_confirmed`、`training_admitted`）、`pending` 数，以及适用的人工语义、框和掩膜确认计数。适用确认维度尚未完成时保留 pending；不适用的人工确认维度明确写“不适用”，不得用程序化或空缺字段代替。

## 权威源

| 事实 | 权威源 |
| --- | --- |
| 识别任务清单 | config/perception_tasks.json |
| 数据任务与目录 | config/workspace_layout.yaml |
| 训练目标 | config/task_targets.yaml |
| 组合轴 | config/combination_axes.yaml |
| 全量 train 规模与来源组成 | config/full_train_allocation.yaml 与 datasets/scripts/<任务>/config/dataset.yaml |
| 父子源、split 与 80% 上限 | datasets/scripts/<任务>/config/source.yaml 的 pairing |
| 可执行计划 | datasets/scripts/<任务>/config/*.json |
| 对象类别、所有者与权威 | config/workspace_ontology.yaml |

## 规则

- 新需求项统一使用 TASK-RNN；不再新增语义 ID 或裸数字 ID。
- 语义 ID 和裸数字 ID 是冻结别名；保留原 ID 和原含义，修改文字不重新编号。
- 新增配对设计需求用 PAIR-RNN，新增校核规则用 PAIR-VNN；现有 R1-R12、V1-V12、BSL 编号冻结。
- 任何 ID 改动先更新引用；跨文档引用必须指向已登记 ID 或生成锚点。
- 计数必须写单位；图片、实例、父组、状态、预算、容量、配额不得互相替代。
- 计划数量不是实际产量，设计比例不是实测分布，未知不得写成零或成功。
- 标注事实按 source_available、annotated、candidate、auto_validated、human_confirmed、training_admitted 六个独立层计数；摘要同时展示六层、pending 和适用的人工语义/框/掩膜确认计数。
- pending 是未完成或未确认状态；缺少 accepted_by、未审核、未准入或无法判定不得自动改写为 rejected/excluded。
- 自动或程序化验证只填 auto_validated，不填 human_confirmed；人工语义确认与人工框、掩膜确认分开记录。
- rejected 和 excluded 只有在存在实际失败证据时使用，不得用“人工接受为 0”代替“标注或标签为 0”。

## 章节序号

长设计或流程文档可用编号 H2；生成文档和短说明不强制编号；跨文档引用使用锚点，不写裸“第 N 节”。

## 维护

```bash
python lab/scripts/contracts/id_policy.py --write
python lab/scripts/contracts/id_policy.py --check
```

本文只校验编号和口径声明，不判断需求证据是否充分、计划是否能执行、数据是否已生产或模型效果。
