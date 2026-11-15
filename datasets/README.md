# 数据区

> 本文提及的部分本地素材、数据和历史运行产物未随本仓库提供；相关名称保留用于说明来源和实验背景。

`datasets/exports/` 按任务组织，每个 `datasets/exports/<任务>/` 下只放当前数据集的 `train/`、`test/` 和 `val/` 三个集合。`datasets/samples/<任务>/` 放该任务的当前小样；`datasets/scripts/<任务>/` 放该任务的数据集与小样生产入口。每任务只维护一个当前小样和一个当前数据集，错误或过期结果按用户决定清除；代码和配置历史由 Git 管理，生成来源、状态和参数记入现有清单。已有旧目录的迁移、引用及回执处理按实际核实；目录存在不代表生成入口已接入。原始素材范围及记录职责见[原始来源定位](#原始来源定位)；标注和审核候选位于 `annotated/`。各任务开始生产前须按实际文件、清单和工具确认输入。

## 标注状态

标注、审查和训练准入按 `source_available`、`annotated`、`candidate`、`auto_validated`、`human_confirmed`、`training_admitted` 六层独立计数；pending 不是 rejected／excluded。详细定义、merge-preserve 写回要求和程序化／人工确认边界见[标注与准入状态层](../docs/DATASET_PRODUCTION.md#标注与准入状态层)。

| 范围与单位 | `source_available` | `annotated` | `candidate` | `auto_validated` | `human_confirmed` | `training_admitted` | pending |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 人物直接审查记录 | 34 条 | 23 条 | 34 条、72 框 | 23 条（16 正、7 负、46 正例框） | 语义 3；框／空场 0；掩膜 0 | 0 | partial 11 条、26 框；语义 31；标签 34；框／空场 34 条、46 正例框；训练 34 |
| CarRealPicture 标签范围 | 31 张图 | 126 条图像—任务记录 | 126 条记录、94 个框 | unknown | 完整记录语义 0；框／空场 0；掩膜 0 | 0 | 人工复核 126；`partial` 2、`unknown` 1 |
| SiteCapture 五小样 | 41 张图 | 66 条有标签文件记录 | 82 条记录、64 个框 | unknown | 语义 0；框／空场 0；掩膜 0 | 0 | 人工复核 82；partial 类记录 16 |

表中样例是当前可核对的局部范围，不是八任务总量；其余任务未建立统一的六层计数时写 unknown，不从目录、回执、计划或人工确认 0 推断标签为 0。`human_accepted: false`、`accepted_by: null` 或自动判定 `unknown` 均只表示 pending，只有实际失败证据才能记 `rejected`／`excluded`。

目录契约以 `config/workspace_layout.yaml` 为唯一源；结构检查用 `python lab/scripts/contracts/workspace_layout.py check`，根 README 目录表与[数据集工作目录](../docs/DATASET_DESIGN.md#数据集工作目录)的任务表由 `lab/scripts/contracts/sync_layout_docs.py` 生成。训练目标与组合轴分别记录在 `config/task_targets.yaml` 和 `config/combination_axes.yaml`；覆盖与全量容量缺口见[训练目标与组合轴覆盖](../docs/DATASET_DESIGN.md#训练目标与组合轴覆盖生成)。对象类别、所有者、权威源和版本策略见 `config/workspace_ontology.yaml` 与 `docs/WORKSPACE_ONTOLOGY.md`。

每任务 `datasets/scripts/<任务>/config/dataset.yaml` 记录目标与状态并引用本任务 `config/source.yaml`；后者记录来源查询范围和清单，由任务生成器消费。贝叶斯、透视等共同能力放在 `datasets/scripts/common/`，各任务保持专属逻辑。八任务当前入口已读取各自来源配置，实际消费字段和支持分支见[脚本入口说明](scripts/README.md)。

当前生产、渲染、抽帧、打包和训练暂停；八任务来源配置、首批输入及恢复入口已完成本轮代码收尾，文档和配置维护可继续。当前状态、首批预算和后续预计见[生产记录](../docs/DATASET_PRODUCTION.md#当前生产进度)，执行命令见[恢复生产入口](scripts/README.md#恢复生产入口)。每任务使用固定小样和数据集目录，`runs/` 内层 V3 模型及运行记录保持原位；代码、配置和文档历史由 Git 管理，素材与生产回执保留实际来源及用途。

`blender_source_registry.sqlite3` 是 Blender 源资产分类、生命周期、路径别名和已核实关系的权威记录；也保存明确标记为 manifest 声明或未核 claim 的证据记录。由 `lab/scripts/blender/blender_source_registry.py` 明确写入，不受媒体索引同步影响。manifest/JSON/JSONL 是生成时证据，能证明清单当时写了什么，不单独确立当前资产状态或关系；登记库中的 claim 也不自动等同于已核实关系。`annotated/media_index/query_cache.sqlite3` 是从 JSON 分区同步得到的可重建媒体观察查询库；`annotated/media_index/production_state.sqlite3` 保存生产回执。后两者分别用于媒体查询和生产复用，不承担 Blender 资产现状的权威登记。数据库不监控磁盘变化，路径存在性和来源索引新鲜度需按需核验。

**当前入口**：[CarRealPicture 核心车载批次](CAR_REAL_CAPTURE_20261112.md)提供 V4 真实画面依据；当前素材与来源候选见[当前素材与来源候选](#当前素材与来源候选)。车牌沿用省赛模型，不生产车牌数据集；其他八项按[数据集设计](../docs/DATASET_DESIGN.md)逐项制作，协作入口见[总控提示词](DATASET_COORDINATION_PROMPT.md)和[协作协议](DATASET_COORDINATION.md)。数据准备、训练效果与现场验收分别记录。

`raw/` 放原始采集，`annotated/` 放标注与待审核候选。`exports/<任务>/train/`、`exports/<任务>/test/` 和 `exports/<任务>/val/` 是正式集合入口，目录本身不代表训练准入。详细规则见[数据集设计](../docs/DATASET_DESIGN.md#数据集工作目录)。图像、标签及模型大文件默认不入 Git。训练以合成数据为主，同时纳入直接实拍，实拍应占相当比例；缺少标签的当前实拍按任务重建标注，本轮已有标签按实际状态使用；合成包括实拍或数字图片的图像处理合成、Blender 渲染及渲染图与实拍混合。本轮依据现有资料和逻辑一次性制作，每模型仅一次训练，不以模型验证或最终测试为前置；未测效果保持未知。各任务以 30% 直接实拍、40% 实拍背景合成、30% Blender 母场景背景作为初始计划；2026-11-15 用户要求 fire、building_name_paper、waste_bin_state 采用 20/40/40，waste_sticker、meter 采用 10/40/50，person、e_bike、sign 保持 30/40/30。具体组成按来源和标签状态调整，比例不是硬门槛，详见[训练计划](../docs/DATASET_DESIGN.md#八任务训练计划)和[父子源配对](../docs/DATASET_SOURCE_PAIRING.md)。样本分别记录来源谱系和生成方式；实拍拟合渲染与直接相机采集分开统计。已知同源复用及变换记录在现有清单，具体约束见[制作细则](../docs/DATASET_PRODUCTION.md#多原件取样与场景混合)。同图多任务复用时，各模型的标签、类别表与审核状态分别保存，原图与父来源可共享。[逐模型标注规则](../docs/DATASET_PRODUCTION.md#每个模型的标注范围)。

`annotated/person/` 按来源记录区分人物派生贴图、车辆遮挡素材、实拍抽帧及标签、来源观察和历史计划；审查观测已迁 `annotated/common/review/person/`，不在该目录。角色入口集中在 `scripts/person/config/source.yaml` 的 `role_records`；标注保留原位。`mixed_detection_candidates_v2_20261110/manifest.jsonl` 的365条记录已改为明确的工作空间相对路径，并记录对应原清单及记录号，继续共享相邻旧目录的图片和标签。生命周期中22条失效当前状态已改为missing，4条失败/隔离状态保留并标记路径缺失；历史交接保留原声明并明确为快照。此次检查涉及目录、清单及路径，不是逐图重新审核标签质量。

跨任务共用的 Blender 场景放在 `assets/blender/shared/`；背景图、标注候选与来源证据仍放在 `annotated/mixed_scene/` 和原批次清单中。`assets/person/` 是 2026-11-15 由原 `datasets/person/` 迁入的实拍派生与历史遗留素材，与 `raw/person/teammate_submission_20261110/` 是同一批 1,334 个文件、333,540,321 字节的重复副本（抽样 SHA256 相同，不是独立采集）；含 281 张背景帧、1,001 张 synth 图（train 700／val 150／test 150／preview 1）、reports、20 张素材和 7 个旧脚本，保留作只读候选来源，不在此写入新的生成产物。

背景审查属于跨任务能力：任务级计划与判定在 `scripts/<任务>/config/review_plan.json` 和 `scripts/<任务>/config/review_selection.json`（入 Git）；观测度量、叠加图与批次清单统一写在 `annotated/common/review/<任务>/<批次>/`，透视模板导出到 `annotated/common/perspective/<任务>/<批次>.json`。`annotated/` 整体被忽略，不可重算的决定不得只留在其中；判定目前只由自动判据给出，进入 `auto_validated` 而不进入 `human_confirmed`，视觉复核方法与验收口径未定。`accepted_by` 为空只记 pending；缺少人工接受、`unknown` 或尚未准入不能自动转成 `rejected`／`excluded`。审查或手工文件重跑必须 merge-preserve，默认校验只读；模板或整份手工记录重建须显式请求、检测非空人工内容并先备份。判据、放置区域杂物门禁、判定语义和命令见[数据集工作目录](../docs/DATASET_DESIGN.md#数据集工作目录)与[脚本入口说明](scripts/README.md#背景审查入口)；b4/b5 测验及当前 B6 阈值和分发状态见[生产记录](../docs/DATASET_PRODUCTION.md#当前生产进度)。

跨任务背景与来源总账由 `scripts/common/source_ledger/` 重算，报告写入 `annotated/common/sources/ledger.json`；它汇总父组、split、审查状态、共享背景池和 Blender 场景用量，是派生视图而非新来源权威。配对准入以 `python lab/scripts/contracts/source_pairing.py --check` 为单一 checker；source_ledger 的诊断不是第二套 gate。权威与边界见[背景与来源总账](../docs/BACKGROUND_SOURCE_LEDGER.md)。

## 原始来源定位

V4 的原始图片、视频及数字素材来自 `datasets/raw/`，也允许来自 `Archive/Assets/`；Blender 场景和模型允许来自 `datasets/assets/blender/`。标注、manifest、数据库、代码及由允许源派生的中间结果按各自职责保留，不因位于其他目录被当作原始素材或自动删除；其他历史或当前原始源按证据报告。具体选源、处理链和查询边界见[数据集设计](../docs/DATASET_DESIGN.md#数据来源与生成方式的定义)。

按采集批次、原件和派生关系判断来源；任务目录及 `Archive/Assets/` 的内容分类只用于查找可见对象，不代表独立场次、唯一任务或 train／val／test 归属。同一原件可为多项任务提供画面或背景。Archive 原件保留原位，可按入口使用 `datasets/raw/` 下工作副本或被允许的 Archive 原件；19 张 PPM 工作图已无损转为 PNG，原始 PPM 仍在 Archive。[来源副本映射](source_copy_map.json)记录路径前缀、数量和扩展名转换；`source_copy_manifest.jsonl` 逐文件记录原件、副本及已核实的重复关系，供媒体索引查询。复制或格式转换不产生新的独立采集。

[raw 来源与用途候选](annotated/raw_source_classification_manifest_20261114.jsonl)覆盖本机 4,954 张图片，10 张来源类型未知；部分已识别类型的精确拍摄批次仍未知。人物目录另有 281 张[背景视觉候选](annotated/person/background_source_manifest.jsonl)、对应的[定性视角记录](annotated/person/view_geometry_manifest.jsonl)和 1,059 张[物料及旧合成图观察](annotated/person/person_material_visual_manifest.jsonl)；旧标签和集合声明保留证据位置，未自动视作当前训练真值。场景的区域候选和任务观察可通过[媒体查询](../docs/DATASET_DESIGN.md#媒体记录与查询设计)检索，视觉候选与完整目标标签、可直接贴合的区域分别判断。

人物背景的[地面局部观察](annotated/person/ground_surface_manifest.jsonl)覆盖 281 张，其中 223 张有局部地面参考；[接地点候选](annotated/person/placement_ground_visual_manifest.jsonl)中 182 张有条件候选、99 张为未知。候选仅供判断局部接地点，目标尺寸、完整轮廓、遮挡、透视和截断尚需单独核验。

[桶参考图几何](annotated/waste_bin/bin_reference_geometry_manifest.jsonl)覆盖 14 张用户提供的多视角与彩色桶照片，记录桶体和可辨开闭状态候选，以及真实正面分类标志下方的局部贴合候选；侧面、背面与顶部没有推断正面可贴区域。候选坐标、尺寸上限和观察依据以清单为准。

| 来源域 | 当前原件位置与核对范围 | 来源边界 |
| --- | --- | --- |
| 国赛场地实拍 | [Archive 图片和视频索引](../Archive/technical/materials/ASSETS_INDEX.md)对应 83 张按画面主体分类的图片、38 段视频；工作副本按[映射](source_copy_map.json)放入 `raw/e_bike/`、`raw/fire/`、`raw/waste_bin/`、`raw/sign/`、`raw/traffic_light/`、`raw/meter/`、`raw/plate/`、`raw/person/` 和共享的 `raw/site_archive_legacy/` | 用户 2026-11-15 确认全部取自国赛真实场地，同一场地、不同拍摄批次；`raw/*/archive_assets_legacy/` 是 Archive 原件的 `working_copy_of` 工作副本，`legacy` 只表示副本命名，不表示旧场地或过期。各视频的具体场次和独立关系未证实。`pics.zip` 中 14 张 PPM 与分类目录及 `密集人群/` 中的 5 张副本按同一原件管理，后者转成 PNG 工作图也不计新采集。 |
| 小车路线批次 | [CarRealPicture](CAR_REAL_CAPTURE_20261112.md)的工作副本位于 `raw/car_real_capture_20261112/`：18 张任务点 PNG、2,773 张连续帧 JPEG、1 段 MP4、1 份时间 CSV；`train/` 的 516 张是逐帧目录的重复子集 | 同批导出，任务点照片与视频的逐帧对应未建立；`train/` 文件夹名不是当前集合分配。 |
| 场地补录批次 | [补录批次](SITE_CAPTURE_20261112.md)保存 160 张车载 PNG、2 段 MP4；[原件路径清单](annotated/site_capture_20261112/source_inventory.csv)的 162 条当前路径和字节数均已核对 | 160 张跨任务照片位于 `raw/site_capture_20261112/onboard_camera_images/`，目录名指拍摄设备；两段视频的设备与相互场次关系待核。 |
| 物料和专项输入 | `raw/person/team_assets/`、`raw/fire/team_assets/`、`raw/plate/` 保存图案或参考图；`raw/waste_bin/` 包含实拍、外来图标和队员工具包；`raw/meter/` 含仪表视频及两张参考图 | 图案、外部参考、工具包、相机实拍分别标记，不因同在 `raw/` 就视为同类独立采集。 |

`raw/e_bike/archive_assets_legacy/` 保存原电动车与人物目录的 7 张 PNG 工作图和 12 段 MP4 视频，人物任务共用这一副本，不再复制第二份。`raw/fire/videos/train/`、`val/`、`test/` 是预留目录，不代表视频缺失；楼宇视频工作副本位于 `raw/fire/archive_assets_legacy/楼宇/`。`raw/thermal/` 当前为空，上述素材未发现红外热图和逐层热真值；这项是真实输入缺口，不能用 D 楼可见光照片代替。

## Blender 源资产与派生数据

`assets/blender/` 是当前 Blender 模型和可编辑场景的集中位置，源模型按 `shared/` 与任务名分组，导出批次场景快照按 `batch_scenes/<任务>/<批次>/` 分组。源模型、任务工作副本、试验分支与批次快照用途不同，分别沿用原批次清单记录；历史版本保留在 `Archive/technical/`，不并入当前资产树。模型与批次场景二进制不入 Git。迁移保留模型包同目录资源，并更新现有 manifest 的快照位置；新增模型或批次场景的入口写入该集中目录。不要仅凭文件名日期或 `latest` 判定权威版本。

路径职责按素材生命周期区分：`assets/blender/shared/` 与 `assets/blender/<任务>/` 存放当前引用的母场景、任务工作副本、试验分支和待核备份，目录位置本身不表示唯一权威或验收状态；`assets/blender/batch_scenes/<任务>/<批次>/` 只存该批次保留的可编辑场景快照，图像、标签和批次清单按当前数据设计放置；有明确替代、退役或复原依据的历史模型/场景放在 `Archive/technical/revisions/` 或 `Archive/technical/datasets/`，与当前生成输入隔离；临时 Blender 测试夹具留在 `tmp/`，不作为生产来源，也不当作长期归档。

`.blend` 是场景/模型来源，不是训练样本或独立相机采集。渲染、合成、标签和集合分配属于派生记录；同一场景的不同视角或随机渲染仍关联同一源资产。`blender_source_registry.sqlite3` 记录当前资产分类、生命周期、路径及已核实的来源/派生关系；原批次 JSON/JSONL manifest 保留生成时的原始声明和记录索引，是登记关系的证据来源，不是第二份当前状态权威。关系可回指原 manifest、`record_index` 和生成器；未核实关系明确标为 claim 或待核，不按文件名补造。更换源版本不回写旧样本的生成时谱系。工作副本或预览只有经过逐任务核对并满足复用需要后才可纳入长期源资产位置。

数据库当前登记 102 条 Blender 资产记录，现存 101 个文件：`datasets/assets/blender/` 下 58 个当前模型及批次场景快照、`Archive/technical/` 下 42 个归档模型/场景/备份，以及 `tmp/` 下 1 个来源待查夹具；另有 1 条已移除文件的保留记录（`waste_sticker_animated_v4_before_shader_fix.blend`，生命周期 `removed`，按用户要求于 2026-11-14 删除，原路径留作历史证据）。2026-11-14 工作区文件扫描（排除 `.git/`、`.venv/` 和 `node_modules/`）发现 101 个 `.blend`/`.blend1` 文件，全部位于上述三个目录。24 个当前 `.blend` 模型已逐项录入分类及现有证据；34 个批次快照按快照角色登记，42 个归档文件按归档位置登记，临时夹具为未决。归档说明明确列出的 33 个历史修订模型原路径已作为 SQLite 别名登记；父偏移错误的仪表分支标为 `retired` 并保留原路径别名。三个历史批次场景的旧导出路径和场景到批次关系也已登记。仪表静态备份及三份 arena `.blend1` 均归档为恢复候选，保留原路径及与当前 `.blend` 的比较关系；直接保存谱系未证实。除已有明确退役关系及这些候选关系外，其他归档模型的逐版本父子关系没有足够证据时仍保持未登记。

历史检查曾记录 34/34 个批次快照由导出 `manifest`/`run` 的 `editable_preview_scene` 或 `scene_copy` 字段指向；对应数据库关系保留原证据文件和 JSON 字段路径。历史清单另声明 34 条 Blender 场景来源（火灾 6 条 `source_scene`、仪表 12 条 `source_blend`、垃圾桶 16 条 `scene_file`），垃圾桶清单还声明 12 条 `bin_source` 模型输入。另有火灾楼名纸条目录中的 3 份 sidecar manifest 指向仪表模型并描述仪表 Pose，登记为 `unverified_cross_task_manifest_claim`，未并入火灾批次来源。历史上位于 `datasets/exports/` 的原清单多数已随该目录清空而不可按原路径回查；人物的 8 个 2026-11-14 候选批次在 `runs/person/migrated_exports_20261114/`，其清单仍写旧的导出路径，见[制作细则](../docs/DATASET_PRODUCTION.md#多原件取样与场景混合)。登记库中的历史声明和证据路径仍保留，不自动证明当前文件存在或模型物理拟合。原路径缺失时按登记记录和其他现存证据说明范围，不能把历史声明当成独立验证。

```powershell
python lab/scripts/blender/blender_source_registry.py list --state review_pending
python lab/scripts/blender/blender_source_registry.py show blender:datasets/assets/blender/meter/v4_parametric_station_v2_20261113/station.blend
```

首次显式盘点可运行 `python lab/scripts/blender/blender_source_registry.py bootstrap --confirm-inventory`；该操作只新增未登记文件或刷新仍为盘点基线的记录，不重置人工状态。用 `import-batch-manifests --confirm-bounded-import` 可仅针对已登记的 34 个批次场景，按其对应 manifest/run 明确导入快照、批次目录和来源字段声明；不会把其他目录里的同名 sidecar 当成中心化快照关系。单项登记或更正使用 `register`，登记迁移前路径别名使用 `alias`，追加关系使用 `relate`，更正关系使用 `correct-relation`（保留资产事件）。退役应更新生命周期状态并保留记录，不删除历史资产行。媒体索引的 `sync` 不写入此数据库。

当前场景尚未统一：`assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend` 是用户确认的新生产母场景，火焰、电动车、仪表和垃圾桶的多个入口引用；它不是所有任务的统一权威。人物 `render_person_arena_v4.py` 默认 `--scene original` 加载 `arena_indoor_detailed.blend`；显式选择 `user-master` 才加载人物工作副本。标准人物试跑 `v4_user_master_pilot_20261113` 请求 8 张、产出 0 张，均因截断、过小或重叠需复核而拒收；独立近景试跑 `v4_user_master_pilot_near_20261113` 有 8 张候选，语义复核待完成。两种试跑状态分别记录，不能概括为人物已切换或完全未产出。车牌与禁行牌的部分背景混合集成入口仍用 `arena_indoor_detailed.blend`；`arena_indoor_detailed.blend` 源自 `arena_indoor_calibrated.blend` 的外观拟合，不是实测标定。

当前状态分类如下。表中模型路径已集中到 `assets/blender/`；代码或清单引用说明模型当前用途，不自动证明模型、相机或真值已通过验收。

| 状态 | 当前证据 | 生命周期处理 |
| --- | --- | --- |
| 共享母场景候选 | `assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend` 经用户确认用于新生产，已有火焰、电动车、仪表、垃圾桶多个入口引用。 | 保留原位；逐任务比对后再确认长期共享资产资格。 |
| 任务工作副本 | `assets/blender/person/arena_person_user_master_working_20261113.blend` 与 `assets/blender/plate/v4_arena_vehicle_20261113/arena_plate_working.blend` 逐字节比较均与母场景相同，且仍被专用生成入口引用。 | 暂作为任务边界副本保留；待决定生成器是否需要独立路径及其编辑生命周期，不因重复删除。 |
| 仪表任务模型源与实验分支 | PBR v2 `assets/blender/meter/blender_9011_pbr_v2_20261110/meter_9011_pbr.blend` 是后续站房原型的构建输入和父源，保留。V4 渲染及背景标注入口当前使用 `assets/blender/meter/v4_parametric_station_v2_20261113/station.blend`（当前任务候选），其 `mother_blend` 指向 outer-tick 原型。`render_meter_candidates.py` 默认源为 `assets/blender/meter/blender_9011_station_v4_tilt65_640_preview/meter_9011_power_meter_station.blend`（65°，使用 v1 端点）；多个仍存在的导出 manifest 指向 `assets/blender/meter/blender_9011_station_v4_outer_tick_preview_20261112/meter_9011_power_meter_station.blend`（65°，使用 v2 端点）。两场景文件长度相同但逐字节比较不同。对应生成端点可见性及与实物刻度的对应仍未知。 | 按入口和批次分别保留；不凭“最新”名称选唯一版本，不据此认定 Pose 真值已确认。 |
| 废桶任务源与派生候选 | `assets/blender/waste_bin/blender_pbr_v2_20261110/waste_bin_pbr.blend` 仍是 `apply_photo_bin_markings.py` 的父场景，也被多个状态批次清单记录为 `mother_scene`。`assets/blender/waste_bin/blender_photo_markings_v4/waste_bin_photo_markings.blend` 从该场景派生；其 manifest 状态为 `candidate_requires_visual_review`，仅替换红、绿实拍标识，蓝、灰保留；`render_arena_waste_v4.py` 仍引用该模型。Blender 5.2.3 检查到 PBR v2 和照片标识候选的四张 768×768 类别贴图已打包；候选还打包了 2275×1279 的来源照片。两份场景均无外部 `.blend` 库链接。贴入照片的再分发许可条款尚未确认。`assets/blender/waste_bin/waste_sticker_animated_v4.blend` 从照片标识场景派生，是垃圾贴纸动画的任务候选（`candidate_review_pending`）；贴纸输出着色器已修复，保存场景相机平移 0.20 与临时渲染 0.16 不同，后者只做过投影核对。 | PBR v2 保留为当前父场景；照片标识版本保留为待目视审核候选，不视作审核通过或停用。 |
| 废桶 PBR v3 未决模型 | `assets/blender/waste_bin/blender_pbr_v3_20261110/` 含 `.blend` 和预览；当前针对代码、配置、清单及文档的检索未发现消费引用，也没有版本 manifest 或生命周期记录。Blender 5.2.3 结构对照显示 v2 有 87 个对象、v3 有 88 个；检查范围内 v3 多出 `PBR front soft strip` 矩形面光，位置 `(0, -3.3, 2.2)`、功率 210 W、尺寸 `3.0 × 0.5`。其他对象名称及已比较的对象变换、网格计数、材质名称、贴图名称/尺寸/打包状态、渲染分辨率和引擎未见差异。v3 四张 768×768 类别贴图已打包，无外部 `.blend` 库链接。现有 v2/v3 预览图已并排查看：两图均为四色桶的正面室内构图，视觉差异不构成可确认的质量改进证据；这不是受控重渲染比较。未比较全部 datablock 参数，因此结构和预览观察仍不能证明 v3 的生成来源；目录名也不能证明它是 PBR v2 的派生版本或试验分支。 | 保留并标为待核；补足来源、用途和版本关系证据前，不归入基线或试验分支，也不删除/退休。 |
| 已归档历史构建基线 | `Archive/technical/revisions/waste_bin/waste_bin_reference_v13_20261109/waste_bin_reference.blend` 是依据实拍及用户测量近似建模的旧基线；深度和车载观察距离未测。无当前生成入口引用，三个现存历史批次和一个归档批次的 manifest 记录该来源。 | 作为历史构建基线归档；历史 manifest 证明批次曾声明使用，不证明它与 PBR v2 的精确版本关系或已正式退役。 |
| 共享场景及未决候选 | `arena_master_user_20261113.blend` 是用户确认用于新生产的场景，多个任务入口及导出清单仍引用。`arena_indoor_detailed.blend` 仍被人物、禁行牌和车牌部分入口引用；其来源记录指向父场景 `arena_indoor_calibrated.blend`，校准记录说明它是车载实拍外观拟合，不是相机/场地实测标定。`arena_indoor_calibrated.blend` 当前未发现活动代码直接消费者，但仍是 detailed 的已记录父场景。`arena_4_2m.blend` 位于 `shared/`，没有直接文件路径消费者或显式来源关系。火焰贴纸 pilot 的门控记录出现同名支撑对象，但对应 `run.json` 声明场景为 `arena_master_user_20261113.blend`；Blender 5.2.3 确认两场景都含该名称的地面对象，因此对象名不能确立候选文件的来源关系。 | 用户母场景和仍有消费者的 detailed 保留。calibrated 作为来源链上的过程资产保留；`arena_4_2m.blend` 保留为未决共享场景候选，其 `.blend1` 已归档为结构对照未见差异的恢复候选；`arena_indoor_calibrated.blend1` 和 `arena_indoor_detailed.blend1` 已归档为恢复候选，差异与证据边界见[技术修订索引](../Archive/technical/revisions/README.md#共享场景候选备份)。不凭文件名推定备份的直接父子谱系。 |
| 火灾楼体模型族 | `assets/blender/fire/blender_buildings_v13/three_buildings.blend` 是当前火灾和楼名纸条入口的场景源。构建脚本会从合并场景另存 A/B/C 单楼文件，但目前已查的入口和六条火灾批次来源声明均指向合并场景；未发现单楼文件的直接生产消费者，也没有精确运行回执证明现存单楼文件由该脚本本次生成。单楼模型保留为生成组件，单独用途待核。构建脚本将 `datasets/raw/fire/team_assets/image1.png` 和 NotoSansSC 字体打包进场景；几何尺寸按照片和用户提供的平均楼体尺寸估算，不是实测 CAD。 | 合并场景按当前用途保留；A/B/C 保留在中心素材区，登记为用途待核的生成组件，不把脚本可生成说成当前被消费，也不据此推断其与合并场景的精确父子谱系。预览、字体及其 OFL 许可按现有关系保留。不得把楼体尺寸称为实测值。 |
| 仪表参考模型与归档静态备份 | `assets/blender/meter/blender_9011_reference/meter_9011_approx.blend` 有动画重渲染操作说明；`face_generated.png` 在代码、配置、文档及现存样本记录中有多处引用，是现有模型、端点配置和样本记录的来源。原 `.blend1` 备份已移至[技术修订归档](../Archive/technical/revisions/README.md#仪表-blender-模型)，保留 512×512 静态状态、相机位置及两针零旋转；当前 `.blend` 为 1024×1024，具有 721 帧相机和指针动画。两文件对象、网格、材质和贴图一致，但直接父子关系及动画是否为唯一差异未证实。`arena_indoor_calibrated.blend1` 与 `arena_indoor_detailed.blend1` 已按差异归档为恢复候选；`arena_4_2m.blend1` 已归档，结构对照范围和谱系边界见[技术修订索引](../Archive/technical/revisions/README.md#共享场景候选备份)。 | 四份 `.blend1` 均归档为恢复候选，均未确认直接父子谱系；该目录组当前不再有原位 `.blend1`。 |

迁移检查时 Blender 5.2.3 LTS 已成功打开归入 `assets/blender/` 的 57 个文件：源模型 23 个 `.blend`，批次场景快照 34 个；迁移后重新打开检查通过。此后新增的 `waste_sticker_animated_v4.blend` 已在 Blender 5.2.3 中重新打开并核对 240 帧输出链接，保存场景的相机平移 0.20 与临时渲染 0.16 不同，后者只做过投影核对。图像资源已打包，没有外部库链接；唯一未打包资源为 Blender 内置字体。`Archive/technical/` 当前有 42 个 Blender 文件：35 个归档修订模型、3 个历史数据批次场景，以及 4 个 `.blend1` 恢复候选；四份 arena `.blend1` 均已由 Blender 5.2.3 打开检查；本轮归档的 v13 桶模型也成功打开，均无外部库或缺失图像。归档说明记录了各修订模型的原路径及恢复候选状态；`dataset-versions.tsv` 三条含场景的批次与磁盘文件数和字节数一致。归档中的 42 个 Blender 文件均为本机忽略文件，`git ls-files` 未收录，不能据 Git 恢复。火灾 v6 没有模型/预览快照、垃圾桶 v9 原目录为空；状态均为未知，不以相邻版本补齐。全工作区扫描还发现 `tmp/fire_building_library_test.blend`：Blender 将其读作含楼体对象和已打包图像的 Library 容器，场景为空；未找到其 manifest 或代码引用，确切生成来源和保留期限缺失，因此暂留 `tmp/`，不把它认作生产来源资产。位置迁移本身不构成验收结论；当前分类和关系以 SQLite 登记为准，状态待核的模型仍未被判定通过或废弃。现有媒体索引解析器不将 `.blend` 的 `source_blend`、`mother_blend` 建为媒体路径证据，34 个可编辑批次场景字段已更新并解析到现存文件；`query_cache.sqlite3` 只能查询已同步的媒体观察及 manifest 引用，不是资产分类权威库。当前资产分类和关系写入 `datasets/blender_source_registry.sqlite3`；媒体索引解析器仍不单独抽取 `.blend` 引用，来源边由登记库关系和 `evidence_ref` 回指。

共享背景从 SQLite 按原视频和镜头组选择。`annotated/mixed_scene/v4_compositing_backgrounds_20261113/manifest.json` 记录 6 段原视频派生的 120 张 640×360 图，每段 20 张，并保留原视频时间戳及镜头组。使用前按任务检查画面已有目标，符合条件的图片再分配用途。抽帧入口为 `lab/scripts/media/collect_v4_compositing_backgrounds.py`，输入未变时复用生产回执。


[2026-11-12 比赛场地补录批次](SITE_CAPTURE_20261112.md) 的原件集中在 `raw/site_capture_20261112/`，照片初筛、分段和待审核抽帧在 `annotated/site_capture_20261112/`。后续使用批次说明关联原件映射与派生记录。


V4 数据按[制作细则](../docs/DATASET_PRODUCTION.md#任务分工与阶段边界)记录 Blender 背景的实际数量、来源组和视角组；每任务及 split 的来源证据保留在已有清单。

## 当前素材与来源候选

V4 制作按[原始来源定位](#原始来源定位)选取当前存在的素材、Blender 资产及相应标注和来源记录。车载路线批次和场地补录分别见 [CarRealPicture 批次记录](CAR_REAL_CAPTURE_20261112.md) 与 [场地补录记录](SITE_CAPTURE_20261112.md)。原始来源分类见 `annotated/raw_source_classification_manifest_20261114.jsonl`，来源副本关系见 `source_copy_map.json` 和 `source_copy_manifest.jsonl`。

任务窗口开始制作时，按当前文件、清单、索引和生成入口核定具体可用输入。视觉候选、AI 观察、旧标签或配置声明均不自动构成完整真值，也不自动取得训练、验证或测试资格。图片、实例、来源组和评估单元分别计数。V4 任务目标、数据来源与切分约束见[数据集设计](../docs/DATASET_DESIGN.md)和[数据制作细则](../docs/DATASET_PRODUCTION.md)。
