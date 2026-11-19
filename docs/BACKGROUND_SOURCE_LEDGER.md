# 背景与来源总账

> 总账是从任务配置和现有记录重算的父组诊断。当前逐图入库结论和可分配资格由 `datasets/annotated/common/sources/frame_environment.sqlite3` 持久化，迁移状态见[背景图入库与复用](DATASET_DESIGN.md#背景图入库与复用)。

## 目的

跨任务背景与来源总账回答以下问题：

- 每个任务声明了哪些父组和数据源，归入哪个 split，承担什么角色，证据在哪里；
- 背景的审查状态允许它用于合成背景还是确认空负例；
- 哪些父组、共享背景池父视频和 Blender 母场景被多个任务复用；
- 哪些声明缺路径、缺证据、split 冲突、超过父组使用上限，或把候选背景当成确认空负例。

它只读任务配置、清单、共享背景池和 Blender 登记库，把结果写到 `datasets/annotated/common/sources/ledger.json`；该路径整体位于忽略的 `datasets/annotated/` 下，可由代码和原记录重算。

## 标注状态与 pending

总账处理的是父组、split、背景审查状态和引用关系，不是目标框标签审计器。六层口径详见[标注与准入状态层](DATASET_PRODUCTION.md#标注与准入状态层)。当前 `source_ledger check` 可见 8 个任务、127 个父组声明，其中 `task_real` 56、`background` 58、`blender` 13；这不证明每条证据路径和标签语义均已核验。

| 层 | 总账中的当前状态 |
| --- | --- |
| `source_available` | 127 个父组声明；路径和清单可访问性按 checker 逐项报告，不能由父组总数推定全部可用。 |
| `annotated` | unknown；总账只读取声明的 `labels.status`，不读取或审计目标标签内容。 |
| `candidate` | 58 个背景父组中，`candidate_empty` 44、`unknown` 10、`contains_target` 4；另有 13 个 Blender 父组记录。 |
| `auto_validated` | unknown；检查码验证结构和声明一致性，不产生目标标签的自动语义验收。 |
| `human_confirmed` | unknown；总账没有人工语义、框／空场或掩膜确认字段。 |
| `training_admitted` | unknown；总账不生成、不消费训练准入决定。 |
| pending | 当时汇总可见 44 个 `candidate_empty`、10 个 `unknown` 和 4 个 `contains_target` 父组声明。是否有待补证的字段及用途，查对应的逐图任务记录。 |

`accepted_by: null` 和 `human_accepted: false` 记录人工确认状态。影响本批用途的 `unknown` 由入库端定位缺项、补机器证据并写回当前资格。父组总账在重算时读取所支持的声明；本批选源及配额直接查逐图资格。明确失败记录带原因及任务、角色与范围。复核文件重跑合并保留标签和人工决定；模板重建流程按[标注权威记录](../.agents/skills/workspace-doc-governance/references/data-authorities.md)处理。[机器判定与写回](DATASET_PRODUCTION.md#机器判定与写回)

## 权威与派生

| 信息 | 权威记录 | 总账中的处理 |
| --- | --- | --- |
| 父组、角色、split 预留、补采占位 | 各任务 `datasets/scripts/<任务>/config/source.yaml` 的 `pairing` 块 | 读取并校验；不修改、不补写 |
| 批次来源、父视频、记录位置 | 原 JSON/JSONL 清单 | 检查路径、记录数和 `record_index` 范围；不复制记录 |
| 共享背景池 | `datasets/annotated/common/sources/background_pool_manifest.json`，旧混合场景池清单提供来源证据 | 按物理 `parent_group` 聚合原路径与工作副本、任务已选帧和各 split 的分配候选 |
| Blender 场景现状 | `datasets/blender_source_registry.sqlite3` | 只读查询是否存在、生命周期和分类；磁盘目录本身不证明状态 |
| 背景审查状态 | 逐图持久数据库的当前资格与所引用的任务标签、机器观测；`pairing.review.state` 是父组级声明 | 父组查询目前按 `pairing.review.state`，另可查询旧共享池分配声明。逐图用途资格按持久库记录查询，`matrix` 的父组数用于诊断 |
| 生产回执 | `datasets/annotated/media_index/production_state.sqlite3` | 总账不读取、不覆盖；生成复用仍由生产守卫负责 |

总账不替代上述任何记录，也不建立第二套来源台账。配置缺少 `pairing` 时，工具不按目录名或文件名推断父组，只记录 legacy 观察。

## 与 `source_pairing.py` 的分工

配对结构、split/action 枚举、负例角色声明、`capture_id`、`plan_alignment` 和路径由 `lab/scripts/contracts/source_pairing.py --check` 校验。BSL 检查码来自派生总账，反映该工具所读的声明及其状态筛选。配额按本任务有效的逐图记录决定。[配对校核边界](DATASET_SOURCE_PAIRING.md#81-计划对齐与-checker-边界)

### 已核实的物理父源别名

2026-11-17 核对 `parking_003` 时，以下两个仍存在的视频文件大小均为 33,633,840 字节，SHA256 均为 `38be0e149cc821e33874150a6b19c4dd92ba81b1610de52ca51b81a859bf4586`，按**一个物理父视频**计：

| 路径 | 当前引用 |
| --- | --- |
| `Archive/Assets/电动车与人物/视频/场地多辆电动车_直立与倒伏混合_较低视角_003.mp4` | 共享池清单 `parent_group` / `parent_video`；旧实拍清单的 `original_path` / `source_group` |
| `Archive/Assets/working_sources/legacy/e_bike/视频/场地多辆电动车_直立与倒伏混合_较低视角_003.mp4` | `e_bike`、`person`、`fire`、`sign`、`waste_bin_state` 等任务的 `source.yaml` 父源路径；旧实拍清单的工作副本路径 |

`source_pairing.py --check` 对当前八任务报 0 个结构／路径错误，但其实现逐任务检查配对条目，不按内容哈希归并路径，也不检查跨任务物理同源。派生 `source_ledger` 目前也把上述两条字符串路径分开显示。编制本视频相关批次时，应把两条路径及其派生帧合并为同一个物理父组审计实际接触和 split；任务内的主 split 仍读取各自 `source.yaml`，逐帧用途资格读取数据库。`e_bike` 的 `tr_bike_003` 已标记 `historical_exposed`、`unassigned/blocked`，此前训练实际用过 8 张；`sign` 的 `bg_ebike_003` 是已有 3+2 条检查接触的技术 test；垃圾桶状态新补的同视频 test 帧也只作已接触路线的回归。此处记录经哈希证实的别名，不扩大独立场次数量。

## 两层 adoption

| adoption | 输入 | 能回答 | 不能回答 |
| --- | --- | --- | --- |
| `pairing` | 任务 `source.yaml` 的 `pairing` 块 | 父组、角色、split、审查状态、补采占位、80% 覆盖参考、派生数据源继承 | 画面语义、标签真值、独立采集 |
| `legacy` | `source.yaml` 的其他路径字段、`current_selection.backgrounds`、共享池和 Blender 引用 | 声明了哪些路径、哪些路径缺失、哪些场景/共享池被引用 | 父组归属、split、角色和父组覆盖核算 |
| `missing` | 缺少或无法读取 `source.yaml` | 该任务无法参与账目 | 其他内容 |

没有 `pairing` 时，`query` 返回 `pairing_not_adopted`，不返回空列表冒充“没有可用背景”。`legacy` 观察不是父组账目，不能据此声称某任务已受 split 门控。

## 命令

在工作空间根目录运行：

```powershell
.venv/Scripts/python.exe -m datasets.scripts.common.source_ledger.cli check
.venv/Scripts/python.exe -m datasets.scripts.common.source_ledger.cli write
.venv/Scripts/python.exe -m datasets.scripts.common.source_ledger.cli query --task person --role composite_background --split train
.venv/Scripts/python.exe -m datasets.scripts.common.source_ledger.cli matrix
```

`check` 只读并打印摘要，`write` 额外原子写出默认路径；两者可用 `--task` 限定任务、`--root` 指定工作空间根目录。`query` 返回指定任务、角色和 split 的可用父组、未准入父组及原因；pending 不作为排除结论。`matrix` 打印任务×类别×角色×split 的父组计数。

退出码：0 表示没有错误（警告按实际输出）；1 表示有错误，或 `--strict-warnings` 下存在警告；2 表示配置无法读取等运行失败。`write` 即使返回 1 也已写出报告，便于同时检查。这是派生报告诊断；配对准入仍以 `lab/scripts/contracts/source_pairing.py --check` 为准。

## 检查码

编号命名空间和计数口径见[编号与计数口径](ID_AND_COUNT_POLICY.md)；下列 BSL 是冻结的派生诊断码，不是准入 gate。

| 代码 | 级别 | 含义 |
| --- | --- | --- |
| BSL001 | error | `pairing` 的版本、状态、policy、composition_policy 或 plan_bindings 结构无效 |
| BSL002 | error | 父组或数据源 ID 缺失或重复 |
| BSL003 | error | 父组/数据源路径不是工作空间相对路径或当前不存在 |
| BSL004 | error/warning | 证据缺 `manifest`、文件不存在、`record_index` 越界或清单结构无法计数 |
| BSL005 | error/warning | split、status、action 或 same-source 列表无效 |
| BSL006 | error | 背景缺有效 `review.state` |
| BSL007 | error | `empty_negative` 角色使用了非 `confirmed_empty` 的背景 |
| BSL008 | error | 派生数据源缺 `parent_group`，或父组未解析 |
| BSL009 | error | 派生数据源的 split 未继承父组 |
| BSL011 | error/warning | 历史账本对父组少于 5 个报 warning、占比超过 `train_parent_cap` 报 error。两者均用于父组覆盖诊断，生产按当前逐图资格编制可行批次。 |
| BSL012 | warning | 非 train 父组通过 `same_source_allowed_in` 进入 train，需按同源解释 |
| BSL013 | error | 补采占位缺 `capture_id`、split 不一致、`save_to` 不匹配或非 `planned` |
| BSL014 | error | `val_test_mode: reserve_only` 下出现 count/quota/budget 类字段 |
| BSL016 | error/warning | Blender 场景不存在或未在登记库中 |
| BSL017 | error/warning | `plan_bindings` 路径无效或计划不存在 |
| BSL018 | warning | 任务未提供 `pairing`，按 legacy 观察 |
| BSL019 | error/warning | legacy 声明或 `current_selection` 背景路径不存在 |
| BSL020 | error/warning | class、kind、role、labels 等枚举值无效，或背景缺 labels |
| BSL021 | warning | 已被拒绝的背景仍出现在父组中 |
| BSL022 | warning | `contains_target` 背景缺少可用于标签的目标状态 |
| BSL023 | error/warning | 共享背景池清单不可读，或已选用池内图片但未声明该池背景组 |
| BSL025 | info | 同一父来源、共享池父视频或 Blender 场景被多个任务使用 |
| BSL026 | warning | 共享母场景的 pairing 未声明 `shared` split |
| BSL030 | error | 任务 `source.yaml` 缺失或无法读取 |

`BSL010`、`BSL015`、`BSL024`、`BSL027`–`BSL029` 当前为保留空号，未分配含义；如需启用，先在 `source_ledger` 实现并登记口径。

检查只覆盖声明的路径和清单，不扫描全部媒体；清单超过 64 MiB 时不解析计数。错误表示声明之间不一致或当前文件不可达，不表示模型效果或标签语义已经被验证。

## 背景可用性规则

`query` 和 `background_candidates` 当前按以下代码规则筛选 `class: background` 的父组声明：

1. 角色必须在该父组的 `roles` 中声明；
2. 请求的 split 必须是 `split.primary` 或列在 `split.same_source_allowed_in` 中；
3. `review.state` 为 `rejected` 时不可用，但缺少 `accepted_by`、`candidate_empty` 或 `unknown` 不能自动写成 `rejected`；
4. `empty_negative` 只接受 `confirmed_empty`；
5. `composite_background` 接受 `confirmed_empty`、`candidate_empty`、`contains_target`；
6. `contains_target` 还要有 `labels.status` 为 `complete`、`partial` 或 `ready`，否则不能作为已标目标背景；
7. 通过 `same_source_allowed_in` 进入的 split 仍是同源使用，不构成独立场次。

这些状态用于解释旧父组查询结果。合成配额按图像和接收任务记录计算。常规底图从合格空间中选取，排除以本任务目标为主体的帧。背景里非主体的任务范围内目标须有完整标签，并随合成变换更新。标签缺项只限制依赖该项的用途。机器证据和人工确认分别记账。[逐图背景可用条件](DATASET_DESIGN.md#标签空间与用途资格)

以下仅描述旧父组诊断按角色计数的口径。现行 80% 用于覆盖和留源提示，合格未分配帧可由批次按父来源选择。旧诊断中：`empty_negative` 只把 `confirmed_empty` 背景计为合格；`composite_background` 计入 `confirmed_empty`、`candidate_empty`、`contains_target`，其中 `contains_target` 仍需可用标签状态；`labels.status` 为 `excluded` 或 `blocked` 的背景不计入合格父组，但这些状态必须有实际失败证据，不能由缺少人工接受、未知或未准入推导。

父组诊断仍用其声明状态计算。查询未命中时回查逐图权威记录。

## 输出结构

`ledger.json` 顶层字段：

| 字段 | 内容 |
| --- | --- |
| `schema_version` | 当前为 1 |
| `generated_at` | 带本地时区的生成时间 |
| `tasks` | 每任务的 adoption、pairing_status、groups、data_sources、captures、observed、selections、blender、shared_pool、plan_bindings、train_cap |
| `shared.background_pool` | 共享池 manifest、记录数、父视频、各任务选用数和是否声明 |
| `shared.blender_scenes` | 场景路径、存在性、登记库记录、引用任务、角色和 split |
| `issues` | BSL 代码、级别、任务、父组和说明 |
| `summary` | adoption、父组类别、split、审查状态、共享池父视频、Blender 场景和问题计数 |
| `authority` | 各字段的权威记录和派生声明 |

报告不写回 `source.yaml`；修正声明应在任务配置中完成，再重新运行 `write`。

## 能力边界

- **不证明画面语义。** 路径存在、清单可读、`review.state` 合法都不代表背景画面一定可合成，也不代表目标标签完整。
- **不证明独立采集。** 同一父视频的帧、裁剪、增强和渲染不增加父组；跨任务复用只按记录报告，不把同源结果当成独立场次。
- **不做 V11 计划消费检查。** 当前没有通用的计划字段把每个 strata 绑定到父组；工具只校验 `plan_bindings` 路径，不能证明某 split 的生成计划只消费了该 split 的父组。
- **不做 V9 逐字节跨 split 检查。** 这需要样本级文件证据；总账只检查父组级 split，不替代媒体索引或批次清单。
- **不做配额或生产。** 工具不分配张数、不生成图像、不写 `samples/` 或 `exports/`，不改变生产守卫状态。
- **只读声明范围。** 不自动全扫数据目录；新增或移动文件后需按媒体索引规则刷新相关分区，再运行总账。

## 当前 adoption

2026-11-15 合并时，八任务 `source.yaml` 均已有 `pairing` 草案（`status: draft`）。当时 `source_pairing.py --check` 报 0 个错误；`source_ledger` 读取 `pairing` 生成父组/split/审查状态报告。person 的 `occl_009` 当时改为 train，`plan_alignment: aligned`；sign/waste_sticker/waste_bin_state/e_bike 的 `plan_alignment: stale` 仍是开放决定，编译计划时不能当作已闭环。`candidate_empty` 上的 `empty_negative` 已清理，BSL007 不再用于这些项。现在的执行起点见[当前生产进度](DATASET_PRODUCTION.md#当前生产进度)。

## 维护

- 目录位置由 `config/workspace_layout.yaml` 的 `datasets.scripts.common`、`datasets/annotated/common/sources` 声明；修改后运行 `python lab/scripts/contracts/sync_layout_docs.py` 和 `python lab/scripts/contracts/workspace_layout.py check`。
- 对象类别 `source_ledger` 由 `config/workspace_ontology.yaml` 定义；修改后运行 `python lab/scripts/contracts/workspace_ontology.py --write` 和 `--check`。
- 工具测试位于 `datasets/scripts/common/tests/test_source_ledger.py`。本页维护来源总账的判定口径；任务生产状态见[当前生产进度](DATASET_PRODUCTION.md#当前生产进度)。
