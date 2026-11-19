# 数据区

## 来源资产已迁出（2026-11-19）

用户已将选定的真实来源、原始数字图和统一 Blender 主工程迁往独立的 `D:/DevForge/Programming/Languages/Python/VSCode/AIAEC-VisualRecognition`。本次实际选取与新位置见新空间迁移说明（相邻工作区 `AIAEC-VisualRecognition/docs/MIGRATION.md`，未随本仓库分发）。这里的数据库、旧配置和历史批次记录保留原声明；其中 `active_in_use`、当前路径或生产状态不能视为迁出后的现状。部分旧入口引用的原件已迁出，本旧工程不再作为这些来源的生产入口。

本轮没有把新根绝对路径写入旧 raw_source_relocations.json 或旧数据库的 workspace-relative 字段。旧合成集合、权重和测试输出留在旧工程；工作副本与未选中的重复原件也未顺手删除。新工程中的 source.json / asset.json 保存实际新位置及原出处。

`datasets/exports/<任务>/` 保存当前数据集的 `train/`、`test/`、`val/` 集合与任务级数据配置（如 `data.yaml`）。`datasets/samples/<任务>/` 保存当前小样，`datasets/scripts/<任务>/` 保存对应入口。确认生成结果有误且本次包含重生成时，重建受影响的小样或 split，保留原件和人工内容。代码和配置历史由 Git 管理，生成来源、状态和参数记入批次清单。[原始来源定位](#原始来源定位) · [从原始采集到导出](../docs/DATASET_PRODUCTION.md#从原始采集到导出)

## 标注状态

标注、审查与训练准入按六层计数。`human_accepted` 和 `accepted_by` 记录人工确认，机器观察与已有标签分别保留。`pending` 说明证据尚不足，`rejected` 和 `excluded` 需要明确失败依据。历史小样的计数在各自批次记录追溯，当前正式导出查生产进度。[标注与计数口径](../docs/ID_AND_COUNT_POLICY.md#标注状态层) · [当前生产进度](../docs/DATASET_PRODUCTION.md#当前生产进度)

目录契约以 `config/workspace_layout.yaml` 为唯一源；结构检查用 `python lab/scripts/contracts/workspace_layout.py check`，根 README 目录表与[数据集工作目录](../docs/DATASET_DESIGN.md#数据集工作目录)的任务表由 `lab/scripts/contracts/sync_layout_docs.py` 生成。训练目标与组合轴分别记录在 `config/task_targets.yaml` 和 `config/combination_axes.yaml`；覆盖与全量容量缺口见[训练目标与组合轴覆盖](../docs/DATASET_DESIGN.md#训练目标与组合轴覆盖生成)。对象类别、所有者、权威源和版本策略见 `config/workspace_ontology.yaml` 与 `docs/WORKSPACE_ONTOLOGY.md`。

每项任务的 `config/dataset.yaml` 定义目标与状态，`config/source.yaml` 声明来源、角色和 split。实际消费哪些字段、哪些入口已接线，查[脚本入口](scripts/README.md)。数据从录入到正式导出的顺序见[生产流程](../docs/DATASET_PRODUCTION.md#从原始采集到导出)。

| 数据记录 | 用途 |
| --- | --- |
| `annotated/common/sources/frame_environment.sqlite3` | 逐图保存图像身份、捕获身份、各任务标注及空间证据引用和用途资格。资格可查询为 `eligible` 后进入该用途的分配池，已入库、已分集与已导出各自计数。 |
| `annotated/media_index/query_cache.sqlite3` | 可重建媒体索引。用于素材发现和来源反查，更新以同步快照为准。 |
| `annotated/media_index/production_state.sqlite3` | 持久生产回执。与对应批次清单核对实际输入和产物。 |
| `blender_source_registry.sqlite3` | Blender 资产的当前路径、旧路径别名、生命周期和已核实关系。通过 `lab/scripts/blender/blender_source_registry.py` 查询与维护。 |

**当前入口**：[当前生产进度](../docs/DATASET_PRODUCTION.md#当前生产进度)提供各任务的接续点；[CarRealPicture 核心车载批次](CAR_REAL_CAPTURE_20261112.md)提供 V4 真实画面依据，素材与来源候选见[当前素材与来源候选](#当前素材与来源候选)。车牌沿用省赛模型，其余八项见[数据集设计](../docs/DATASET_DESIGN.md)和[数据制作 Skill](../.agents/skills/dataset-production-loop/SKILL.md)。任务负责人及共享写入见[协作协议](DATASET_COORDINATION.md)。

`raw/` 保存任务目标和共享背景工作副本，`annotated/` 保存标注、候选与审查证据。图像、标签和模型大文件按仓库约定保存在 Git 之外。同一原图可供多个任务使用，各任务独立记录标签范围、类别和资格。实拍、实拍增强、实拍底图合成与 Blender 渲染按实际生成方式和父来源分别入账，设计比例与各任务缺口见[训练计划](../docs/DATASET_DESIGN.md#八任务训练计划)。

跨任务共用的 Blender 场景保存在 `assets/blender/aiaec_dataset.blend`，来源证据随原批次清单保存。`assets/person/` 保留队员人物交接包，含 281 张背景帧、1,001 张合成图、标签、图案、报告和旧脚本。原 `raw` 重复树的 1,334 个文件已与保留包逐路径、逐 SHA-256 核对。现用入口引用保留包，原路径关系见[迁移记录](raw_source_relocations.json)。

同一帧且内容、坐标一致的原件和工作路径共享 `frame_id`。同一捕获的不同图像表示用各自 `frame_id`，沿 `capture_identity` 关联。原清单和 `annotated/common/review/<任务>/<批次>/` 保留证据，按需更新受影响的逐图资格。[逐图入库与资格查询](scripts/README.md#逐图入库与资格查询)

跨任务背景与来源总账由 `scripts/common/source_ledger/` 重算，报告写入 `annotated/common/sources/ledger.json`，用于查看父组、split、审查状态、共享背景池和 Blender 场景用量。配对声明用 `python lab/scripts/contracts/source_pairing.py --check` 检查；实际用途和源证据查相应计划及批次清单。[背景与来源总账](../docs/BACKGROUND_SOURCE_LEDGER.md)

## 原始来源定位

`raw/` 的顶层只允许八个任务目录和 `shared/`。每个任务目录只允许 `targets/`，共享目录只允许 `backgrounds/`。任务目标只放选用的数字图案、照片、帧、裁片和渲染工作图，以及实际生成所需的字体及配套许可。共享目录只放背景工作图。完整采集批次和父视频进入采集归档，历史交接包、外部工具包和参考资料按职责保存。来源清单、标注、审核和媒体索引归入 `annotated/`。布局检查会报告 `raw/` 固定层级中的额外入口。保留契约要求的空角色目录，供选用素材时写入。

完整采集批次及实拍参考集中在 [captures](../Archive/Assets/captures/)。旧原件的可追溯工作副本放在 [legacy](../Archive/Assets/working_sources/legacy/)，数字物料与附属工具包放在 [digital](../Archive/Assets/working_sources/digital/)，尺寸、产品与 PDF 裁页参考放在 [reference](../Archive/Assets/working_sources/reference/)。车载工作批次保留原导出的目录结构及 `train/` 抽样副本关系。现用路径在来源清单中更新，旧路径、捕获身份和父源沿[迁移记录](raw_source_relocations.json)追溯。

八项任务的标注与目标源工作副本统一放在 `raw/<任务>/targets/`。`targets` 表示任务输入用途，完整实拍负例也可放在这里并按任务标签计入 `empty_negative`。`digital/` 保存选用的数字物料工作副本，`camera/<采集批次>/{photos,frames,cutouts}/` 保存实拍及可复用裁片，`rendered/<母场景>/` 保存可复用渲染图。副本通过 `source.yaml` 的 `target_working_manifest` 关联原件、帧身份、父视频或母场景及该任务的标签。多任务副本共用捕获身份。源视频、原始数字素材和可编辑 Blender 场景分别关联各自的采集、物料或资产目录。

可跨任务复用的共享背景工作图放在 `raw/shared/backgrounds/{camera,rendered}/`。各任务通过 `source.yaml` 的 `background_pool_manifest` 关联来源，使用逐图库的当前用途资格。放置区域、已有目标与标签状态分别记录。配额先按实拍父源或 Blender 母场景归组，再选帧和视角。旧混合场景清单、几何审查及生产回执继续提供证据。[背景与来源总账](../docs/BACKGROUND_SOURCE_LEDGER.md)

V4 的生成入口可读取 `datasets/raw/` 的工作副本及允许使用的 `Archive/Assets/` 原件。可编辑 Blender 场景及模型集中在 `datasets/assets/blender/`。来源及派生关系以各任务来源配置和实际批次清单为准。具体选源、处理链和查询边界见[数据集设计](../docs/DATASET_DESIGN.md#数据来源与生成方式的定义)。

按采集批次、原件和派生关系判断来源。任务目录及 `Archive/Assets/` 的内容分类可用于优先发现跨任务底图候选。接收任务按实际画面判主体、目标有无、标签完整性和用途。独立场次与 train／val／test 归属按捕获和使用记录确定。同一原件可为多项任务提供画面或背景。Archive 原件保留原位，可按入口使用 `datasets/raw/` 下工作副本或被允许的 Archive 原件。19 张 PPM 工作图已无损转为 PNG，原始 PPM 仍在 Archive。来源副本、编码转换和重复关系见[来源副本映射](source_copy_map.json)及 `source_copy_manifest.jsonl`。[标签、空间与用途资格](../docs/DATASET_DESIGN.md#标签空间与用途资格)

[raw 来源与用途候选](annotated/raw_source_classification_manifest_20261114.jsonl)覆盖本机 4,954 张图片，10 张来源类型未知；部分已识别类型的精确拍摄批次仍未知。人物目录另有 281 张[背景视觉候选](annotated/person/background_source_manifest.jsonl)、对应的[定性视角记录](annotated/person/view_geometry_manifest.jsonl)和 1,059 张[物料及旧合成图观察](annotated/person/person_material_visual_manifest.jsonl)；旧标签和集合声明保留证据位置，未自动视作当前训练真值。场景的区域候选和任务观察可通过[媒体查询](../docs/DATASET_DESIGN.md#媒体记录与查询设计)检索，视觉候选与完整目标标签、可直接贴合的区域分别判断。

人物背景的[地面局部观察](annotated/person/ground_surface_manifest.jsonl)覆盖 281 张，其中 223 张有局部地面参考；[接地点候选](annotated/person/placement_ground_visual_manifest.jsonl)中 182 张有条件候选、99 张为未知。候选仅供判断局部接地点，目标尺寸、完整轮廓、遮挡、透视和截断尚需单独核验。

[桶参考图几何](annotated/waste_bin/bin_reference_geometry_manifest.jsonl)覆盖 14 张用户提供的多视角与彩色桶照片，记录桶体和可辨开闭状态候选，以及真实正面分类标志下方的局部贴合候选；侧面、背面与顶部没有推断正面可贴区域。候选坐标、尺寸上限和观察依据以清单为准。

| 来源域 | 当前原件位置与核对范围 | 来源边界 |
| --- | --- | --- |
| 国赛场地实拍 | [Archive 图片和视频索引](../Archive/technical/materials/ASSETS_INDEX.md)对应 83 张按主体分类的图片与 38 段视频，来自同一场地的不同批次。工作副本见[路径映射](source_copy_map.json)。 | 按采集批次登记父来源。原始 PPM、分类目录原件和转换 PNG 用 `working_copy_of` 关联，按对应来源计数。 |
| 小车路线批次 | [CarRealPicture](CAR_REAL_CAPTURE_20261112.md)的工作副本位于 `Archive/Assets/captures/car_real_capture_20261112/`，含 18 张任务点 PNG、2,773 张连续帧 JPEG、1 段 MP4 和 1 份时间 CSV。 | `train/` 是原导出的 516 帧抽样子集。当前已核实的照片、帧及时间关系见[点位对齐](annotated/car_real_capture_20261112/waypoint_video_alignment.json)。 |
| 场地补录批次 | [补录批次](SITE_CAPTURE_20261112.md)保存 160 张车载 PNG、2 段 MP4；[原件路径清单](annotated/site_capture_20261112/source_inventory.csv)的 162 条当前路径和字节数均已核对 | 160 张跨任务照片位于 `../Archive/Assets/captures/site_capture_20261112/onboard_camera_images/`，目录名指拍摄设备；两段视频的设备与相互场次关系待核。 |
| 物料和专项输入 | 人物、火焰、车牌图案及垃圾贴纸工具包归 `Archive/Assets/working_sources/digital/`。仪表产品参考和人物尺寸图归 `Archive/Assets/working_sources/reference/`。桶开闭现场参考及仪表父视频归 `Archive/Assets/captures/`。 | 图案、外部参考、工具包和相机实拍分别保留来源。 |

`Archive/Assets/working_sources/legacy/e_bike/` 保存电动车与人物共用的 7 张 PNG 工作图和 12 段 MP4 视频。楼宇视频工作副本位于 `Archive/Assets/working_sources/legacy/fire/楼宇/`。红外热图和逐层热真值仍是来源缺口。

## Blender 源资产与派生数据

### 单文件编辑与生产接线

唯一现行 Blender 工程为 `datasets/assets/blender/aiaec_dataset.blend`。Blender 自动生成的 `.blend1` 是编辑备份，生产入口统一读取 `.blend`。母场景保留共用场地和基准布置；共用模型及可复用小物件在同一文件内按 Collection 管理。不同任务用同文件的 Scene、相机和任务控制对象组织，复用同一个共享 Collection。任务独有的相机轨道、贴纸位置和摆放变化保存在任务侧。共享资产的通用改动在所属 Collection 保存，任务内的布置用本地控制对象或参数表达。

2026-11-16 合并检查曾重新打开统一工程，核对到 9 个 Scene，均未链接外部 `.blend`。`Arena` 的 `AEC Arena` 是共享场地，XY 为地面、Z 向上，场地坐标以米计。`AEC_eight_camera_routes` 保留八套 CAM、PATH 和 RIG。`Arena Waste Sticker Preview` 在同一文件内引用这两个共享 Collection，通过本地 `Waste Sticker Preview Camera` 和 `Waste Sticker Placement` 实例预览 `AEC Waste Stickers`。贴纸根对象和类别动画在 `Waste Stickers` 修改，预览构图和实例摆位在预览 Scene 修改。重开后核对了八路相机及四色桶贴纸的 240 帧类别。

**现用渲染入口与场景**

| 用途 | 读取的 Scene 与 Collection | 入口 |
| --- | --- | --- |
| 人物、电动车场景 | `Arena` 为母场景分支，`Arena Detailed` 保留详细场景分支。 | `render_person_arena_v4.py` 的 `user-master`／`original`，`render_e_bike_blender_batch.py` 的 `S_master_candidate`／`S_indoor_detailed` |
| 楼名纸条 | `Buildings`，含 A/B/C 楼子集合和四台相机。 | `render_blender_names.py` |
| 火焰 | `Fire Buildings`，共用楼宇几何并独立控制 Collection。 | `render_blender_fire_dataset.py` |
| 桶开闭 | 当前前景生产使用 `Waste Bins Photo` 的 `AEC Waste Bins Photo`，盖板从桶后垂下。 | `render_blender_waste_bin.py` |
| 垃圾贴纸（桶面分支） | `Arena` 及来自 `Waste Stickers` 的 `AEC Waste Stickers`。 | `render_sticker_arena_v4.py` |
| 温湿度表 | `Meter Station` 为 Z-up 米制开放前景场景，旧房间默认关闭，输出透明 RGBA；九基础机位各有仰视、平视、俯视，共 27 个预设。安装倾角由 `Station dial tilt pivot` 控制；场地分支复用同一模型。后台入口恢复计划使用的局部坐标。[场景、相机与快照](scripts/meter/README.md#blender-编辑场景) | `render_meter_candidates.py`、`render_meter_arena_v4.py` |

使用 Blender 的任务入口通过各自 `source.yaml` 或共享选择器定位统一工程。后台渲染通过 `blender_workspace.py` 在内存选择 Scene 和所需 Collection，计划和回执按 `文件#Scene或Collection名` 追溯内部来源。名称调整时更新依赖它的实际任务引用。图像、标签和批次清单仍放任务的输出位置。[场景选择实现](scripts/common/blender_workspace.py) · [任务命令](scripts/README.md#恢复生产入口)

#### 共享 Blender 底图选源

新批直接查询逐图库中目标任务、split、用途均匹配的 `composite_background=eligible` 帧，按可放条件和批次预算安排渲染与合成。通用 Arena 母场景的同一张渲染图可以服务多个任务，每个任务保存自己的标签和用途资格。实拍数量用于描述来源覆盖和评估范围。既有两张实拍桶面等固定选图数字对应各自历史批次，来源与产量以原回执为准。

2026-11-17 核对的五批 Blender 底图均已入库。下表按任务列出这五批内具备 train 合成底图资格的帧数。

| 底图 | 当前可用任务和帧数 | 源清单 |
| --- | --- | --- |
| `Arena`、`Arena Indoor Soft Light`、`Arena Detailed`，每场景 192 张，由八条轨道各取 24 帧 | 楼名纸条、仪表各 576 张。人物、电动车、禁行牌、桶开闭各 564 张有合格接地区。 | [Arena](annotated/common/sources/aiaec_eight_routes_20261117/manifest.json) · [Arena Indoor Soft Light](annotated/common/sources/aiaec_eight_routes_indoor_20261117/manifest.json) · [Arena Detailed](annotated/common/sources/aiaec_eight_routes_detailed_20261117/manifest.json) |
| `Buildings` 的 C 楼无火焰立面 | 火焰 24 张，每张 10 个经遮挡核对的可贴区域。 | [C 楼立面](annotated/common/sources/aiaec_fire_c_facade_20261117/manifest.json) |
| `Arena Waste Sticker Preview` 的无物品贴纸四色桶面 | 垃圾贴纸 24 张，共 96 个可贴桶面。 | [桶面](annotated/waste_bin/waste_sticker_arena_surfaces_20261117/manifest.json) |

纯 Arena 空场同时提供八任务完整空负。火焰和垃圾贴纸合成分别使用带可贴立面和桶面的任务底图。五批图均关联同一个 Blender 工程父来源，按原始帧、相机、场景和工程版本追溯。批次在[共享背景池清单](annotated/common/sources/background_pool_manifest.json)登记。固定选源入口接入这些资格的情况见[当前生产进度](../docs/DATASET_PRODUCTION.md#当前生产进度)。

`Waste Stickers | Front scan camera` 是活动相机，在第 1–240 帧依次扫过左、中、右三个方位和远、中、近三档距离。镜头位于桶口上方约四分之一个桶身高度，沿水平方向拍摄。时间轴的九个 `Orbit` 标记与后台读取的固定机位对应。四张物品贴在每个整数帧分别取图案、位置、缩放与转角，等比缩放范围为原始纸片的 1.10–1.42 倍。位置从整个桶体正面取样，纸片四角留在桶面内，贴到固有分类标志处时按实际遮挡出图。后台比较机位时可固定贴纸帧。例如在工作区根目录执行以下命令。图像、掩膜和候选框写入该帧的空小样子目录。

```powershell
& $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/waste_sticker/render_sticker_arena_v4.py -- --orbit-frame 5 --out datasets/samples/waste_sticker/orbit_frame_0005 --samples 8
```

在 Blender 的 `Waste Stickers` 场景中，展开大纲视图里的 `AEC Waste Stickers`，勾选蓝、红、绿、灰四个桶集合中需要显示的 1–4 个。集合复选框控制整个桶及其贴纸。后台固定帧预览、逐状态出图与实拍合成见[脚本入口](scripts/README.md#垃圾贴纸按可见桶数生成与合成)。

`Waste Bins Photo | Front scan camera` 是活动相机，在第 1–120 帧经过同一套机位。`--bins` 接受类似 `blue=closed,red=hanging_back` 的逐桶姿态列表，省略的桶会隐藏。`--compare-lids green` 在每个机位各渲一张绿桶开盖和闭盖图，盖板打开时完全翻至桶后。检测框根据实际可见的整桶几何生成，盖板或桶口掩膜提供状态标注证据。轨道诊断、合成前景和缩影总览保存在 `datasets/annotated/waste_bin_state/`。当前数据缩影使用 `datasets/samples/waste_bin_state/` 下的 `images/`、`labels/` 和 `manifest.json`。

```powershell
& $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/waste_bin_state/render_bins_photo_orbit.py -- --out datasets/annotated/waste_bin_state/photo_orbit --visibility-probes
& $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/waste_bin_state/render_bins_photo_orbit.py -- --out datasets/annotated/waste_bin_state/photo_orbit_lid_comparison --compare-lids green
```

楼名字体和仪表贴图打包在工程内，素材区的工作副本用于来源核对。`Buildings` 的四台楼名相机、低机位球面轨迹及逐楼开关见[楼名入口](scripts/building_name_paper/README.md#blender-光照轨迹和楼体开关)。火焰镜头独立保存于 `Fire Buildings`，场景分别保留自己的时间轴、环境与 Collection 开关，共享楼宇物体及网格。跨任务同时出图见[并行渲染与工程写入](#并行渲染与工程写入)。

2026-11-16 已从真实入口验证楼名、火焰、垃圾贴纸、桶状态、仪表和电动车的两个场景，检查运行至生产守卫之前。另核对人物两个场景和摆桶模型取源。验证范围为渲染前检查，实际出图按批次清单与回执核对。

相机轨道微调、贴纸挪动和物件位置修正直接保存到同一工程。运行时由任务参数覆盖的值改对应参数。更新后核对受影响任务的相机、取源和样本，批次回执保留实际读取的工程及配置版本。需要复现具体批次时，快照归 `Archive/technical/datasets/<任务>/<批次>/`；被替代且有恢复价值的基线归 `Archive/technical/revisions/<任务>/`。归档前核对仍被引用的文件、资源与清单，保存原路径和关系。日常保存沿用原文件和资产身份，已有清单保留生成时的来源声明。

`.blend` 是场景/模型来源，不是训练样本或独立相机采集。渲染、合成、标签和集合分配属于派生记录；同一场景的不同视角或随机渲染仍关联同一源资产。`blender_source_registry.sqlite3` 记录当前资产分类、生命周期、路径及已核实的来源/派生关系；原批次 JSON/JSONL manifest 保留生成时的原始声明和记录索引，是登记关系的证据来源，不是第二份当前状态权威。关系可回指原 manifest、`record_index` 和生成器；未核实关系明确标为 claim 或待核，不按文件名补造。更换源版本不回写旧样本的生成时谱系。工作副本或预览只有经过逐任务核对并满足复用需要后才可纳入长期源资产位置。

### 并行渲染与工程写入

1. **每个渲染批次启动一个后台 Blender 进程。** 所有进程可以读取同一个已保存的 `datasets/assets/blender/aiaec_dataset.blend`。调用前明确任务、Scene、所需 Collection、计划和输出。入口在各自进程内选择场景并设置帧、相机、灯光和任务对象。空场批次可读取 `Arena`，人物批次可同时读取 `Arena` 并在自己的进程里加入人物立牌。交互式 Blender 会话由正在编辑工程的人使用。
2. **每个输出目录只有一个写者。** 并发批次的输出目录须符合各自入口的约定，图像、掩膜、标签、清单和临时输出的路径互不重叠。同一任务和 split 若竞争同一输出目录，依次运行。各批次使用自己的计划和生产回执。
3. **按机器容量安排进程数。** 启动前核对渲染设备、显存和内存。部分入口默认启用全部匹配的 GPU，按入口实际支持的设备选择安排任务。容量不足时减少同时运行的进程，把余下任务排队。某个批次失败时先查该进程的日志和回执，再处理该批次的输出并重试。
4. **工程的保存由一个写者串行完成。** 生成器在内存里摆放物件、改机位和渲染。生产守卫会在批次结束时核对输入文件，依赖当前工程版本的批次结束后再保存新版。轨道、模型和共享 Collection 的长期改动由编辑者保存到统一工程，之后启动的批次读取新版本。

### 资产盘点与迁移记录

2026-11-16 将原活动树的 60 个 Blender 文件按原字节归档，34 个批次快照进入 `Archive/technical/datasets/`，另 26 个文件进入 `Archive/technical/revisions/` 对应任务或 `arena` 的 `blender_sources_20261116/`。归档文件均从新位置重开，SHA-256 与迁移前一致。63 个配套文件随历史场景归档。统一工程随后移至 `blender/` 根目录，原 `shared/` 下的工程路径保留为登记库别名。[历史修订索引](../Archive/technical/revisions/README.md) · [历史批次索引](../Archive/technical/dataset-versions.tsv) · [配套文件迁移记录](raw_source_relocations.json)

旧共享场景、任务工作副本、楼体、桶和仪表版本用登记库查旧路径、当前归档路径及证据。历史批次清单保留生成时的路径。车牌三个历史重放脚本与仪表历史可见性审计通过归档读取相应旧文件。归档和登记可用于复现与追溯，场景几何和物理真值仍按当时的验证范围解释。被 Git 忽略的场景文件从本机归档恢复。

```powershell
python lab/scripts/blender/blender_source_registry.py show datasets/assets/blender/aiaec_dataset.blend
python lab/scripts/blender/blender_source_registry.py list --state review_pending
# 用原资产 ID 查询已归档的版本
python lab/scripts/blender/blender_source_registry.py show blender:datasets/assets/blender/meter/v4_parametric_station_v2_20261113/station.blend
```

登记新资产使用 `register`，迁移旧路径使用 `alias`，追加或更正已核实关系使用 `relate` 或 `correct-relation`。更新退役资产的生命周期并保留事件。资产路径和用途由登记库维护，媒体索引同步只更新素材查询快照。

## 当前素材与来源候选

从[原始来源定位](#原始来源定位)查工作副本和父来源，从[当前生产进度](../docs/DATASET_PRODUCTION.md#当前生产进度)查实际产物。车载路线和场地补录分别见[车载实拍](CAR_REAL_CAPTURE_20261112.md)和[场地补录](SITE_CAPTURE_20261112.md)。按任务、用途和 split 从逐图库查询当前可用素材，历史视觉候选查原批次证据。[选源与查询命令](scripts/README.md#逐图入库与资格查询)

旧共享背景清单 `annotated/mixed_scene/v4_compositing_backgrounds_20261113/manifest.json` 保存 6 段原视频派生的 120 张 640×360 图、原视频时间戳及镜头组。需从视频补采时运行 `lab/scripts/media/collect_v4_compositing_backgrounds.py`，再按当前任务的目标标签和空间用途入库。Blender 背景的源资产、视角和实际用量随批次清单记录。
