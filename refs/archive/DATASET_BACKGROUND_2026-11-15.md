# 2026-11-15 背景审查与批次记录

这是文档镜像提交 [7563ac4](https://github.com/101AoiKoori/AIAEC-DevStudio-Open/blob/7563ac420dbb3bf445433057a36a71a07ecb211b/docs/DATASET_PRODUCTION.md) 中的阶段记录。正文中的“当前”“尚未执行”和“暂停”均指该段落写下时的情况。B6 及其派生结果保留作历史。

## 当时可见的标注层

下表按现有结构化记录核对；`unknown` 表示当前没有可引用的该层记录，不等同于 0，也不表示被排除。计数日期为 2026-11-15。

| 范围与单位 | `source_available` | `annotated` | `candidate` | `auto_validated` | `human_confirmed` | `training_admitted` | pending |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 人物直接审查记录 | 34 条记录 | 23 条有完整标签记录 | 34 条记录、72 个候选框 | 23 条记录：16 正例、7 负例、46 个正例框 | 语义 3；框／空场 0；掩膜 0 | 0 | partial 11 条、26 个候选框；语义 31 条；标签确认 34 条；框／空场确认 34 条记录、46 个正例框；训练准入 34 条 |
| CarRealPicture 当前标签范围 | 31 张图 | 126 条图像—任务记录 | 126 条记录、94 个候选框 | unknown | 完整记录语义 0；框／空场 0；掩膜 0 | 0 | 人工复核 126 条；`partial` 2 条、`unknown` 1 条继续 pending |
| SiteCapture 五个小样清单 | 41 张图 | 66 条有标签文件的图像—任务记录 | 82 条记录、64 个候选框 | unknown | 语义 0；框／空场 0；掩膜 0 | 0 | 人工复核 82 条；`partial` 14 条、`candidate_partial` 2 条 |

人物分层以 `datasets/scripts/person/config/direct_annotation_layers_20261115.json` 为准；CarRealPicture 标签由 `evaluate_field_capture.py audit-labels` 从两个标签文件读取；SiteCapture 小样由五个 `review_manifest.json` 读取。自动审查判定、生产回执和模型评分均不自动改写这些层。

记录日期：2026-11-15。人物任务已实跑三批：`exports/person/train` 3000 张、`val` 300 张、`test` 300 张，各含图片、YOLO 标签和 manifest，生产守卫已落三份回执（`planned_samples` 3000／300／300）。`samples/person` 另有 36 张最小链路试验批。其余七任务当前产量仍为 0，渲染、抽帧、打包和训练继续暂停。当前阶段已完成来源选择、可用标注用途、人物身份区域及遮挡次序、仪表和桶状态渲染计划；缺标注或用途未确认的素材保留 pending，只有具有明确失败证据的任务角色用途才排除，原件和原始记录保留。

人物三批由 `datasets/scripts/person/build_full_person_plan.py` 编译，计划为 `config/full_train_plan_3000.json`、`val_plan_300.json`、`test_plan_300.json`，出图入口为 `compose_planned_ground_person.py`。train 组成：1650 无遮挡单人、375 直立电动车遮挡、375 倒伏电动车遮挡、600 双人；val/test 各 40 无遮挡单人、100 直立遮挡、100 倒伏遮挡、60 双人。三批共用 18 张身份纹理、4 张遮挡片（来自 2 段电动车视频）和 3 种色调；train 用 4 个 legacy 空场背景，val 留第 5 个，test 用 site_capture 来源组。已知缺口：三批都没有 zero-person 空场负样本，`config/task_targets.yaml` 中 person 的 counts 含 `empty` 但本批未覆盖；val 与 train 的背景来自同一条视频 `空场与场景物料_001.mp4`，只有 test 更换来源组，因此 val/test 只能作换帧、换布局的泛化证据，不能作跨场景或跨身份泛化证据。3600 条记录的 `source_role` 均为 `unknown`（`historical_exposure_incomplete`），manifest 状态为 `candidate_not_semantically_verified`，只能记 candidate 层，未经人工确认，也未准入训练。

同日解除三处生产死锁：任务输出目录（`samples/<任务>`、`exports/<任务>/{train,val,test}`）尚未生成时 `workspace_layout.py check` 只报提示，由生产守卫在生成时创建，不再要求预先存在；`compile_background_dispatch.py` 不再要求 `accepted_by` 等于固定人名，改为要求非空人工接受；计划型生产仍禁止 adoption，无回执的中途产物通过删除该输出目录恢复，删除后重新生成由入口负责。

当前任务配置位于 `datasets/scripts/<任务>/config/`：`dataset.yaml` 记录总目标，`source.yaml` 记录选定输入、父来源、原清单定位和排除项，各计划 JSON 记录入口能执行的状态及配额。任务路径、必需配置和标注根以 `config/workspace_layout.yaml` 为准，结构检查用 `python lab/scripts/contracts/workspace_layout.py check`。具体命令、参数、输出路径和工序顺序集中在[恢复生产入口](../../datasets/scripts/README.md#恢复生产入口)。本文件维护阶段与数量预期；旧批次和文档修订从 Git 历史查询，不作为当前产量或恢复入口。

来源用途与首批计划已接入现有媒体查询库：8 条任务配置声明、13 份现存计划共 1,725 个状态及其路径反查引用，桶状态另有 1 份待编译计划。2026-11-15 重建声明分区并全量同步后，查询库为 `ready`、223 个分区，快照 2026-11-15T12:13:34Z，`missing_sources` 25。`datasets/assets/person` 分区记录 1,316 张图，只有媒体条目，没有清单或标签证据，也没有任务配置引用它；它与已不存在的 `datasets/person/` 按显式迁移关系登记，新旧路径在 `path_history` 中关联。生产回执库和 Blender 登记库未修改。配置及计划可按任务、集合、字段角色或输入路径查询，命令见[数据库输入查询](../../datasets/scripts/README.md#从现有数据库取得当前输入)。用途声明、待生成前景和实际媒体分别记录，生产继续暂停。

人物候选审查已完成三批：`v4_review_20261115`（8 帧，2 ready／6 unknown／0 excluded，提交于 `5b5fac8`）、`v4_review_20261115_b2`（18 帧，3／15／0，提交于 `03b35f4`）和 `v4_review_20261115_b3`（16 帧，6 ready／10 unknown／0，2026-11-15 07:15 提交）。b2 的 `metrics.jsonl` 已在 07:09 用同一代码重算，按当前判据是 4 ready／14 unknown，与提交时的 3／15 不同；度量目录里的结果可由代码和原图重算。b3 的 10 个 unknown 均为 `geometry_rejected`，其中 mixed_scene 桶背景 5 张全部落在该类；11 条记录带 `source_unresolved`，表示旧实拍的父视频关系未核实。`review_selection.json` 在 b1–b3 阶段只保留最近一批，旧批可从 Git 历史核对；当时 `accepted_by` 为空。当前 B6 状态见下文；旧 `min_patch_free_ratio`、`max_patch_obstacle_components` 仍作为默认关闭的兼容预筛。三批观测已从 `annotated/person/` 迁到 `annotated/common/review/person/`，透视模板在 `annotated/common/perspective/person/`；审查判据、产物和父源筛查属于跨任务 common，人物只是第一个使用者。

2026-11-15 人物跨分支小样另有独立验收记录：当前 6 项候选覆盖直接实拍 3、实拍合成 1、Blender 2。5 项已由用户完成图片级目标与遮挡语义接受，并固化到 `datasets/scripts/person/config/toolchain_review_acceptance.json`；另有 1 项 Blender 原始预览按场景图和实例 ID 掩膜确定性生成，以 `programmatic_blender_generator` 接受，不计作人工验收。当前 6 项均为 `pass_candidate`。独立标签审计为 5 条，即 4 条完整正例和 1 条确认负例、10 个候选可见实例；负例帧已通过全帧检查，3 条实拍正例框仍标作 `Codex visual review` 候选，Blender 预览的 3 个框来自程序化生成证据。该层确认不自动等于通用人工逐框真值或训练准入；单帧预览也不是数据集批次。自动背景审查三批中的 `accepted_by` 空值只约束那三批审查，不覆盖本次任务配置的人工接受记录或程序化渲染验收。

2026-11-15 放置区域杂物门禁由 `datasets/scripts/common/review/placement.py` 实现并在 b4 测验中启用。b4 从 `query_cache.sqlite3` 的 281 条 person 背景候选路径中按排序等距选 8 张，加入 b3 的 11 张非桶回归帧，共 19 帧；结果为 4 ready、3 excluded、12 unknown。`bg_00232`（对象占用 0.179、最大组件 0.167、垂直 0.0125、自由矩形 0.200）、`frame_007001`（0.341、0.249、0.0097、0.250）和数据库选出的 `bg_00240`（0.248、0.205、0.0265、0.117）判 `excluded`（`placement_patch_cluttered`）；`bg_00086` 对象占用 0.0991 接近 0.10 阈值，按临界带判 `unknown`（`placement_patch_unsure`）；其余 11 帧为 `geometry_rejected`。薄长地面标线单独记录 `marking_ratio`，与本任务已有目标框重叠的组件记为 `task_target`，均不计杂物。b4 计划与选择写在 `annotated/common/review/person/v4_review_20261115_b4/`，`review_selection.json` 仍为 b3；b4 是判定测验，不是训练选择。阈值是设计假设，尚未用独立标定集验证，其他任务未启用该门禁。

2026-11-15 地面几何增加回退并在 b5 测验中复测同一 19 帧。主路径 VP patch 失败时依次尝试水平相机 patch（`placement_patch_level_camera`）和 Gabor/FFT 纹理地面顶边（`horizon_texture_fallback`、`placement_patch_texture_fallback`），纹理回退要求最小 patch 深度且地平线不超过 0.85H。b5 结果为 5 ready、5 excluded、9 unknown，`geometry_rejected` 由 11 降至 7。`bg_00040` 由几何失败恢复为 ready（水平相机 patch）；`frame_006001`、`bg_00280` 恢复 patch 后被杂物门禁判 `excluded`；`bg_00200` 恢复 patch 后对象占用 0.1306 仍判 `unknown`；`bg_00122`、`bg_00120` 的纹理回退因地平线过低被几何阶段拒绝，仍为 `geometry_rejected`。b5 计划与选择写在 `annotated/common/review/person/v4_review_20261115_b5/`；该批当时未更新当前选择，属于几何回退测验。回退 patch 的 reason code 保留在 metrics 中；阈值仍是设计假设，尚未用独立标定集验证。
b5 之后对剩余 5 张 `geometry_rejected` 做目视归类：`bg_00165`、`bg_00160` 为办公室场景，画面含桌面、立牌和电动车；`frame_003001` 为立牌与电动车近景；`frame_005001` 为垃圾桶近景；`bg_00080` 为黑色反光墙与窄地面。它们不是干净的任务背景，不再继续几何调优。另做一次只读标线线检测试验：`bg_00086` 的放置区内未检出白线，对象占用主要来自地面纹理/颜色离群，标线修正无贡献；`bg_00232`、`bg_00240` 的对象占用仅分别下降约 0.02、0.01，判定不变。因此不加入标线硬门禁，保留 `marking_ratio` 报告。b4/b5 是测验批次；当前采用的 B6 判定和阈值见下文。

2026-11-15 背景审查 B6 已采用用户接受的 24 帧 clean／clutter／geometry_failed 分类拟合放置阈值；24/24 为该标定集合内的拟合一致数，不是独立泛化测量。B6 对 19 帧给出 6 ready、8 unknown、5 excluded；用户接受 11 条确定判定和全部父视频关系，8 条 unknown 保持 pending。当前 `review_plan.json` 和 `review_selection.json` 指向 B6，公共审查能力仍在 `datasets/scripts/common/review/`。6 条 ready 已写入 `datasets/scripts/person/config/background_dispatch.json`，现有 54 状态、预算 36 的人物 train 计划按 JSON 分发背景；生成入口已通过 54 状态的内存合成检查。36 张实际批次首次因预建空目录被生产守卫拒绝；移除空目录后运行到第 4 张时因旧暗化函数不接受计划的 1.1 亮度增益而停止，留下 3 张图片和 3 个标签、无完成清单。人物入口现已增加有界亮度处理；自动批准审查拒绝了对该中途目录的递归删除命令，故未重跑，不计完成批次。两组父视频在候选状态中为 43／11；11 个 site 状态中至少 8 个必须选用，以使预算 36 的执行结果满足单父组不超过 80%。本阶段已在 `datasets/samples/person/` 实际生成 36 张图片和 36 个标签，清单状态为 `candidate_not_semantically_verified`；父组为 28／8，最大占比 77.8%，生产守卫完成回执已写入。此前 `datasets/exports/person/train/` 的 3 张图片和 3 个标签仍是中途产物，无完成清单。3000 张全量计划和 val/test 尚未执行。

2026-11-15 用户指定 V4 数据制作**先推进人物**；其他任务在人物链路（生成、审查、回执、测试骨架）跑通后按同一入口复用。

2026-11-14 21:43:51–21:51:43 之间 `datasets/exports/` 被重建为空骨架：8 个任务目录和 24 个 `train/val/test` 目录均为该窗口新建，旧批次树不再存在，执行者与命令在工作空间内无记录。这些空目录是代码约定的正式集合输出根（`datasets/scripts/common/paths.py` 的 `split_dir()` 与入口白名单），不是待清理残留。人物候选池 124 条声明输入中 77 条因此失效（person 21、waste_bin 20、fire 12、meter 9、mixed_scene 7、e_bike 3、sign 3、plate 2、annotated/meter 1）；`原根目录工具区的 build_training_packages.py` 使用的 9 个 `data.yaml` 均不存在，该入口已于 2026-11-15 删除，当前没有打包入口。人物 8 个 2026-11-14 候选批次位于 `runs/person/migrated_exports_20261114/`（1,367 个文件），未随该窗口消失。
