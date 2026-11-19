# 八任务父源容量与全量 train 分配可行性

> 2026-11-15 全量 train 草案的容量核对快照。下文的待补采、预留和标签状态基于当日输入。部分来源路径属于当时的盘点位置，现用路径见[迁移记录](../datasets/raw_source_relocations.json)。现行批次以[当前生产进度](DATASET_PRODUCTION.md#当前生产进度)及其实际计划为准。父子源类别和配对结构见[数据集父子源配对与分集预留](DATASET_SOURCE_PAIRING.md)。

## 范围与口径

- **父组**：按父照片、父视频、Blender 母场景或固定物料组计数。同一父视频的帧、裁剪、增强、渲染和不同机位不增加父组；不同文件或不同种子本身不证明独立采集。拍摄场次独立性未知时记录未知，不按文件名推断。
- **已用**按本次估算时 `source.yaml`、首批计划和清单对来源的引用计数，实际产量另记。
- **可用状态**分三级：已审/选定、已登记候选未审、未知。候选池、AI 观察、旧标签和配置声明不自动构成训练真值；空候选不当作确认负例。
- **数据快照**：`datasets/annotated/media_index/query_cache.sqlite3` 对应目录快照 2026-11-15T03:55:15Z，222 个分区、24 个 `source_present=false`（清单与数据区说明一致）。Blender 资产按 `datasets/blender_source_registry.sqlite3`（102 条登记、101 个现存文件）核对。数据目录不入 Git；本报告在独立 worktree 中只读核对本机数据，未改动数据、配置或计划。
- **现行规模**：八任务合计 Train 24,300 张，与各任务 `dataset.yaml` 和 `config/full_train_allocation.yaml` 对照。垃圾贴纸按 1,800 张及 0/30/70 的来源目标编制，仪表保持 10/40/50，其他任务份额见[全量 train 配置](../config/full_train_allocation.yaml)。下文其余任务的来源计数保留 2026-11-15 盘点范围。
- **80% 参考份额**：按任务×角色统计 train 实际使用的父组，并提示可留给 val/test 与后续补量的来源。父组不足 5 个时报告来源集中程度，继续用现有合格素材制作；同源 val/test 标明范围，独立来源结论另用未接触父组。重复帧或增强不增加独立父组数。

## 来源域与父组总览

### 原始来源域

| 来源域 | 已登记父组 | 媒体数量 | 已审/已选 | 限制与证据 |
| --- | --- | ---: | --- | --- |
| `datasets/raw/person/` | 18 身份图案（16 社区 + 2 非社区）；1 个背景批次（父视频 `空场与场景物料_001.mp4`，281 帧）；5 张无视频归档图 | 1,340 图 | 图案 2 张入选首批；背景 23 帧审查 | 背景 `background_source_manifest.jsonl` 为缩略图 AI 候选，不是确认空场；281 帧的父链部分未核 |
| `datasets/raw/e_bike/` | 12 视频（4–011 单车环绕、001–003 多车、012 场地扫描）；7 张车载照片 | 12 视频 + 7 图 | 008/009 两个平视转台视频有精修掩膜 4 张；7 张照片整组预留、不参与 train | 审核帧 97 帧来自 5 视频（`video_review_20261110/manifest.json`）；场次关系 unknown |
| `datasets/raw/fire/` | 楼体照片 67 张（A 25 / B 18 / C 24）+ D 楼 2 张 + 火焰纸片 1 张；楼宇视频 8；图案 2 张（`image1.png`、`实拍.jpg`） | 72 图 + 8 视频 | 3 个楼面裁剪（A_04/B_51/C_33，各 1 张背景照片）；63 个审核候选覆盖 17 个视频（每视频 3 帧）及照片候选 | 裁剪只支持零/单贴；整图多贴分支未审；`real_review_candidates_20261110/manifest.json` 场次 independence unknown |
| `datasets/raw/sign/` | 1 个牌具视频（转面） | 1 视频 | 8 个掩膜视图：6 train / 1 val / 1 test | 单物理牌具；同属一个父视频 |
| `Archive/Assets/captures/waste_bin_state/` 及对应的旧实拍与贴纸物料 | 7 桶视频 + `运动方式.mp4`；用户多视角桶照片 14；归档桶照片 2；固定图案 24；参考图 2；工具包 zip 1 | 8 视频 + 42 图（18 现场/参考图 + 24 图案） | 首批 10 背景（同一父视频）；2 个红桶面几何 | `bin_reference_geometry_manifest.jsonl` 覆盖 14 张用户照片；24 图案 `SourcePic` 齐 |
| `datasets/raw/meter/` | 1 仪表视频；归档照片 2；外部参考 1；PDF 截取 1；CarRealPicture 帧 45 | 1 视频 + 4 参考图 | 3 个背景（`frame_003601/009901/010501`，同一父视频）；12 前景渲染状态 | 57 条视觉记录中无完整 8 点 Pose；`real_meter_visual_manifest.jsonl` |
| `Archive/Assets/captures/site_capture_20261112/` | 2 视频：`full_scene.mp4`（11,378 帧、23 段）、`person_standees.mp4`（1,502 帧、12 段）；160 张车载 PNG | 2 视频 + 160 图 | 多任务共用；车载图像经审查的按任务见各节 | `segments.csv` 35 段；160 图 `target_instances_manifest.jsonl` 观测：person 68、e_bike 60、bnp 21、fire 21、sign 18、meter 7、waste_bin_state 7、waste_sticker 5 |
| `Archive/Assets/captures/car_real_capture_20261112/` | 1 视频；2,773 帧序列 + 516 张重复子集 + 18 张任务点 PNG；52 段 | 3,307 图 + 1 视频 | 任务点正对与连续帧；任务用途候选 | 同批导出；帧与视频对应未建立；`inventory.json` |
| `Archive/Assets/working_sources/legacy/site/` | 9 段三楼与场地视频 + 1 段综合视频 | 10 视频 | 6 个父视频已抽帧为共享背景池 | 同场地不同拍摄批次（用户 2026-11-15 确认）；具体场次关系未证实 |
| `Archive/Assets/` | 原件：3,395 图 + 39 视频（含 CarRealPicture 3,307 图 + 1 视频） | — | 工作副本映射见 `datasets/source_copy_map.json` | 原件保留原位；工作副本不产生新采集 |

### 共享背景池

`datasets/annotated/mixed_scene/v4_compositing_backgrounds_20261113/manifest.json` 从 6 个父视频各抽 20 帧，共 120 张 640×360 图：

| 父视频 | 段数/候选 | 保留帧 |
| --- | ---: | ---: |
| `三栋楼_垃圾桶与路口_003.mp4` | 35 候选、2 排除 | 20 |
| `三栋楼_路口及垃圾桶_005.mp4` | 37 候选、3 排除 | 20 |
| `三栋楼侧向扫描_009.mp4` | 34 候选、2 排除 | 20 |
| `人物立牌正背面_电动车直立_倒伏和遮挡_场地扫描_012.mp4` | 37 候选 | 20 |
| `场地多辆电动车_直立与倒伏混合_较低视角_003.mp4` | 34 候选、4 排除 | 20 |
| `四色桶较低视角_灰绿闭盖_红蓝开盖_004.mp4` | 36 候选、2 排除 | 20 |

上述表格记录旧共享背景基批次。现行按逐图库的任务用途与 split 选源，共享池扩展及实际批次见[当前生产进度](DATASET_PRODUCTION.md#当前生产进度)。

### Blender 母场景

`datasets/blender_source_registry.sqlite3` 中与八任务相关的当前资产：

| 资产 | 任务 | 生命周期 | 证据等级 | 被引用 |
| --- | --- | --- | --- | --- |
| `shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend` | 共享 | active_candidate | approximate | person（工作副本与其逐字节相同）、e_bike、waste_bin_state（`arena_bin_asset`）、waste_sticker、meter；注册表另记火焰入口引用 |
| `shared/arena_4m2_pbr_20261112/arena_indoor_detailed.blend` | 共享 | active_in_use | fit_from_photo | person（默认 `--scene original`）、e_bike、sign/plate 部分背景混合入口 |
| `shared/arena_4m2_pbr_20261112/arena_indoor_calibrated.blend` | 共享 | retained_parent | fit_from_photo | detailed 的父场景，无直接消费者 |
| `fire/blender_buildings_v13/three_buildings.blend` | fire | active_in_use | approximate | fire、building_name_paper |
| `fire/blender_buildings_v13/{A,B,C}_building.blend` | fire | review_pending | approximate | 生成组件，无直接消费者 |
| `meter/v4_parametric_station_v2_20261113/station.blend` | meter | active_candidate | approximate | meter（前景渲染） |
| `meter/blender_9011_pbr_v2_20261110/meter_9011_pbr.blend` | meter | retained_parent | fit_from_photo | 站房原型的父源 |
| `waste_bin/blender_pbr_v2_20261110/waste_bin_pbr.blend` | waste_bin | active_in_use | fit_from_photo | 桶状态；垃圾贴纸场景的父源 |
| `waste_bin/blender_photo_markings_v4/waste_bin_photo_markings.blend` | waste_bin | candidate_review_pending | fit_from_photo | 照片标识候选 |
| `waste_bin/waste_sticker_animated_v4.blend` | waste_sticker | candidate_review_pending | approximate | 贴纸动画场景（从照片标识场景派生） |

批次场景快照 34 个（fire 6、meter 12、waste_bin 16）只证明批次场景，不计新增母场景。每个母场景按一个父组计；同一母场景跨任务多次渲染不增加父组，也不应各自按满额计。

## 八任务逐项容量

### 人物 `person`

train 3,000：空场 300；无遮挡社区 975、非社区 975；电动车遮挡 750（直立 375、倒伏 375，两类身份各约一半）。

| 角色 | 父组 | 可用基数 | 当前已选/已用 | 状态与限制 |
| --- | --- | ---: | --- | --- |
| 背景 | `full_scene.mp4` | 11,378 帧 / 23 段 | 3 帧（40.0、265.2、285.2 s） | 3 张均 `human_accepted: false`；其中 2 张自动判据 unknown；计划用人工地面多边形与足点 |
| 背景 | `空场与场景物料_001.mp4` | 281 帧 | 0 选定；23 帧进审查计划 | 缩略图 AI 候选，「no obvious person」不等于确认负例；父链 10/23 已核、13/23 未核 |
| 背景 | `person_standees.mp4` | 1,502 帧 / 12 段 | 0 | 立牌拍摄视频，可按帧选直接实拍或已标目标背景 |
| 背景 | 共享池其余 5 视频 | 100 帧 | 0 | 人物空场状态未审；004 的 5 帧在 b3 全部 `geometry_rejected` |
| 背景 | CarRealPicture 等 4 个父视频 | 候选池 7,772 条声明、3,389 条现存 | 0 | 审查队列非训练导出；近重复与全图语义审查未完成 |
| 前景 | 18 身份图案（16 社区 + 2 非社区） | 18 | 2（community_1、noncommunity_F2） | 固定物料；身份区域为 AI 设计候选，非测量 |
| 遮挡物 | 008 倒伏、009 直立 | 2 父视频、4 精修掩膜 | 2 | 与 e_bike 共用 |
| Blender | arena master / detailed | 2 场景 | 均被入口引用 | 母场景用户确认；跨任务共享 |

补充事实：`teammate_submission_20261110` 另有 1,001 张旧 synth 图（旧集合声明 train 700 / val 150 / test 150 / preview 1）；JSONL 共 1,000 条，其中 900 条有框。`datasets/assets/person` 下的三份 JSONL 与 raw 三份逐文件同哈希，不双计。旧集合声明与标签保留为历史证据，不自动作为当前训练真值，也不增加独立父组。

当前自动化盘点由 `python -B datasets/scripts/person/inventory_auto_annotations.py` 从未改动的主清单重算：

- 六个直接实拍审查清单位于 `datasets/annotated/person/` 和 `datasets/annotated/site_capture_20261112/person_vehicle_v4_review/`，共 34 个 review 记录。状态分层独立计数为：source/available 34、annotated 23、candidate 34、auto_validated 23、人工语义确认 3、人工框/空场确认 0、training admitted 0。auto_validated 中包括 16 个正例共 46 个可见框和 7 个自动完整负例；另有 11 个 pending partial 保留候选框，未判 rejected/excluded。分层记录在 `datasets/scripts/person/config/direct_annotation_layers_20261115.json`。
- 人工语义确认仅覆盖 `person_000201` 负例和 `person_000401`、`person_000601` 两个正例共 6 个框；其余 31 个 review 记录仍 pending 人工语义确认。人工接受不包含框坐标或掩膜确认，因此 34 个记录的 human box/absence confirmation 均为 0，training admission 也为 0。
- CarRealSequence 的三份实例清单按父图和框去重后，有 132 张人物候选父帧、343 个框，其中 18 个有类别、325 个 `class_id=null`。SiteCapture 的 160 张车载图清单有 68 个 person 记录、152 个框，其中 34 个有类别、118 个类别未知。两边合计 200 个唯一候选父帧、495 个唯一框；全部记录均 `exhaustive=false`，因此仍保留为清单候选，不能直接当作完整 YOLO 标签。
- `v4_review_20261115` 至 `b5` 的 80 条自动度量覆盖 46 张唯一帧；只有 b4、b5 写有选择判定，分别为 4/19 和 5/19 个 `ready`，`accepted_by` 均为空。这些是父帧几何/空场审查度量，不是人物框或 YOLO 标签。
- `mixed_detection_candidates_20261110` 的 659 张是合成/图像处理候选而非直接相机采集，其中 611 张为 teammate synth、48 张为本地车辆遮挡合成。`teammate_real_scene_candidates_20261110` 的 317 张全部是前一批同路径图片的子集，二者并集仍为 659 张，不能相加。

结论：

- **直接实拍 900：仍缺量，但已有 auto_validated 层。** 六个直接审查清单的分层为 source/available 34、annotated 23、candidate 34、auto_validated 23（16 正、46 框；7 负）、human semantic 3、human box/absence 0、training admitted 0；pending partial 11。工具链另记录 1 帧实拍背景合成正例和 1 帧程序化 Blender 原始预览正例。CarReal/SiteCapture 清单中还有 200 个唯一候选父帧和 495 个唯一人物框，但 443 个类别未知且所有帧均未穷举；这些记录保留为 candidate/pending，不能计入 auto_validated 或人工确认。该结论表示已有可用的自动验证层，不表示 900 帧配额或训练准入已完成。
- **实拍背景合成 1,200：受限。** 帧数充足，但已审背景父组只有 2～3 个（<5），实际首批只选 `full_scene` 的 3 帧；281 帧 001 背景未完成逐帧人物空场审查。空场 300 张尤其依赖尚未确认的空背景。
- **Blender 900（上限 1,200）：可行。** 母场景在册且入口引用；标准试跑 `v4_user_master_pilot_20261113` 请求 8 张、产出 0 张（截断、过小或重叠拒收），近景试跑有 8 张候选待语义复核，说明容量存在但画面仍需按任务门槛筛选。共享母场景需与 e_bike、waste_sticker、meter 合并记账。

当时的草案给人物 train 分配 300 张空场。后续按已编译的人物计划实际导出三个 split，背景判定与使用情况见[当前生产进度](DATASET_PRODUCTION.md#当前生产进度)。

### 电动车 `e_bike`

train 4,000：直立/倒伏 × 街道/停车区 × 零/单/多车 12 组，零车 400、单车 1,600、多车 2,000；零车组 target=empty，不带姿态类别。

批内已有完整来源计划 `v4_combination_source_plan_20261114`（记录在 `datasets/annotated/e_bike/v4_car_sequence_review_20261113/manifest.json`）：

| 分支 | 组配额 | 父组 | 可用基数 | 当前状态 |
| --- | ---: | --- | ---: | --- |
| 直接实拍 | 1,200 | D_site_photo / D_car_capture / D_bike_003_video | 575 / 575 / 50 | 全部 `conditional`；只有 4 帧有完整 AI 标签 |
| 实拍背景合成 | 1,600 | G_full_scene（`full_scene` 4 帧）/ G_car_capture（CarRealPicture 6 张任务点照片）/ G_legacy_scene（`空场与场景物料_001` 2 帧） | 400 / 600 / 600 | 仅 G_full_scene 4 帧 `ready`；另两组 0 合格 |
| Blender 背景 | 1,200 | S_indoor_detailed / S_master_candidate | 600 / 600 | conditional；入口为固定 pilot |

补充事实：

- 审核帧 97 帧来自 5 个父视频（008 15、009 12、005 15、003 6、012 49），全部 `pending_manual_review`；另可用的直接实拍候选包括 160 张车载图的 60 条 e_bike 观测和 CarRealPicture 2,773 帧。
- 前景掩膜 21 张候选全部来自 008/009 两个平视转台视频；精修 4 张。`F_field_005_006` 未被现有入口接通。
- 全部 7 张旧车载照片整组 `reserved_before_train`，不参与任何 train 角色。
- 批次报告（`runs/e_bike/authoritative_plans_20261114/usage_batch_20261114/report.json`）记录：不可用的另两组实拍背景份额已重分配给 G_full_scene，12 个空车状态是 4 帧 × 3 色调，不是 12 个父组。
- 街道/停车区标签未写入背景清单，50/50 的位置配额当前无法按来源证明。

结论：

- **直接实拍 1,200：缺口。** 1,150 张无完整真值，现有 4 帧同属一个视频。
- **实拍背景合成 1,600：受限。** 旧快照的 3 个已登记父组中有 1 个合格；前景主体来源 2 个（<5），多车、人物遮挡和画外截断分支入口未接通。
- **Blender 1,200：受限。** 场景与配额已声明，但入口不能消费批量计划，需先实现批量化。

### 火焰贴纸 `fire`

train 3,000：A/B/C 楼面 × 零/单/多贴 = 200×3 + 400×6。

| 角色 | 父组 | 可用基数 | 当前已选/已用 | 状态与限制 |
| --- | --- | ---: | --- | --- |
| 背景 | 楼面照片 | 67 张（A 25 / B 18 / C 24）+ D 楼 2 | 3 张（各楼 1 张：A `另一侧斜视_整体_013`、B `中距离斜视_楼体和格栅_004`、C `正侧斜视_L形窗格立面_006`） | 每张 2 个状态（零/单贴），共 18 张预算；裁剪只支持 0/1 贴 |
| 背景 | 楼宇视频 | 8（A 2 / B 4 / C 2） | 0 直接消费 | 63 个审核候选覆盖 17 个父视频，每视频 3 帧 |
| 背景 | 场景视频 / CarRealPicture | 9 + 任务点图 | 0 | 跨任务背景候选；空场/目标完整性未审 |
| 图案 | `image1.png` | 1 | 1 | 固定唯一图案，`ready` |
| 参考 | `实拍.jpg` | 1 | 0 | 纸边与成像参考 |
| Blender | `three_buildings.blend` | 1 | 1 | 与 building_name_paper 共用 |
| 负例 | 红绿灯正面照等 | 1 | 1 | `hard_negative` 候选 |

结论：

- **直接实拍 600：缺口。** 42 条用途投影中 direct_sample 均 `blocked` 或 `candidate`，无完整整图监督；16 条 unassigned 直接候选、9 条回归候选也不能充当完整正例。
- **实拍背景合成 1,200：受限。** 三张背景照片可承载零/单贴；多贴 1,200 张需要新楼面裁剪或转为 Blender，现配置裁剪的 `supported_counts` 只有 0/1，无法承载多贴。多贴候选照片（如 `正面_楼名与3张火焰纸贴_017`、`正面_楼名和4张火焰纸贴_023`）存在但整图标注被排除。
- **Blender 1,200（上限 1,200）：可行。** 三楼合景在册并被两个任务引用；多贴可在场景中生成。

### 楼名纸条 `building_name_paper`

train 3,000：无纸 1,000、有纸 2,000；单名 1,200、多名 800；单楼/多楼各 1,500。

| 角色 | 父组 | 可用基数 | 当前已选/已用 | 状态与限制 |
| --- | --- | ---: | --- | --- |
| 直接记录 | `full_scene.mp4` | 2 正例 + 3 负例（5 帧） | 5 帧 | 全部完整检测候选；原字串未知不监督 |
| 审核候选 | full_scene、fire 楼宇视频等 | 57 个审核文件 | 0 | 纸条框与文字真值逐帧待核；不可读保持未知 |
| 纸条几何 | CarRealPicture 序列 | 3 个手估纸框（6 条记录） | 0（共享预留） | 手估框，非训练导出 |
| 背景 | 3 栋楼 + full_scene | 3 楼面 + 帧源 | 组合计划 40 张来自 2 个视频父源 | `combination_plan.json` 两条 track，每组 10–30 |
| Blender | `three_buildings.blend` | 1 场景 | 900 状态计划 | 600 单楼单名、150 双楼多名、150 单楼无纸 |

结论：

- **直接实拍 600：缺口。** 仅有 5 帧完整记录（1 个父视频），其余 57 个审核文件未形成完整真值。
- **实拍背景合成 1,200：受限。** 组合计划只有 2 个视频父源、预算 40；1,200 张需要扩背景组并补齐纸条真值；已有纸框 3 个，固定文字真值靠生成字符串。
- **Blender 1,200（上限 1,200）：可行。** 现有 `blender_plan.json` 为 900 张计划容量（900 个状态），覆盖单名、多名、无纸；按 40% 配额需再扩 300 张 Blender 计划容量；与 fire 共用母场景。

### 禁行牌 `sign`

Train 2500、Val 300 已生成并核对，实际框数为2000／275，空标签500／75。使用344／67张底图或原帧、15／4个父来源，背景父组跨集重叠与相同RGB图片均为零；两集均有四视面×远近／截断／杆底座16格。Train含800张Blender底图派生、1700张相机来源；20张直接实拍正例按来源分为9／11。完整来源及旧框继承见[全量清单](../datasets/exports/sign/manifest.json)和[审计](../datasets/annotated/sign/full_dataset_audit.json)。440张已接受小样保留；前景共享同一牌具视频，Val为受控诊断，另已生成Test300张、225框、75空负，使用一个停车视频父组的20帧；该视频已有旧诊断接触，作为回归，尚无独立真实场景和模型效果结论。

2026-11-18 对旧 inventory 的 155 条未知背景完成写回，147 条合格帧已分配为 Train 70、Val 77、Test 0，含 139 张实拍、8 张旧 Blender，10 张包含已有牌具完整框。另 4 条以牌具为主体记录合成用途受阻，4 条落点不明的合成用途为未知，完整无牌标签仍支持负例。新增 14 个实拍父路径，多数仍是同一室内模型场地。原路径数、帧数和场次数按各自证据统计。[155 帧复核回执](../datasets/annotated/common/review/sign_background_155_20261118/recheck_receipt.json)

按既有任务证据保留旧牌具框，编制新牌具的落点、遮挡与图签。分集时读取本批实际父来源和跨任务接触。`bins004` 本批已归禁行牌 Val，与跨任务的既有 Val 接触一致。[Train 小样](../datasets/samples/sign/train/manifest.json) · [Val 小样](../datasets/samples/sign/val/manifest.json) · [全量训练入口](../datasets/exports/sign/data.yaml)

### 垃圾桶开闭 `waste_bin_state`

train 3,500：空场 300；四色 × 开/闭 × 单/多桶 16 组 × 200。

| 角色 | 父组 | 可用基数 | 当前已选/已用 | 状态与限制 |
| --- | --- | ---: | --- | --- |
| 背景 | `full_scene.mp4` | 14 帧审后背景 | 10 train + 2 val + 2 test | 全部同一父视频；val/test 与 train 同源，不是独立父组 |
| 背景（可扩） | 共享池其余 5 视频 | 100 帧 | 0 | 无桶/含桶状态未逐帧审 |
| 桶体 | 4 个物理桶（蓝/红/绿/灰） | 4 | 4（模型与实拍均有） | 固定实物 |
| 桶视频 | 7 + `运动方式.mp4` | 8 父视频 | 0 直接；23 个审核候选（21 帧） | 桶掩膜与逐桶开闭真值待审 |
| Blender | `waste_bin_pbr.blend` | 1 场景 | 19 状态渲染计划 | 16 单桶 + 2 四桶 + 1 空场；可扩四色开闭组合 |

结论：

- **直接实拍 700：缺口。** 23 个候选未形成逐桶开闭真值；现有批次不含可直接入 train 的完整标签。
- **实拍背景合成 1,400：受限。** 10 个 train 背景全部来自 1 个父视频（合格父组 1，<5）；4 个 val/test 背景同源；需扩共享池并完成无桶/桶状态审查才能满足多父来源要求。
- **Blender 1,400（上限 1,400）：可行。** 桶场景在册，19 个渲染状态是已明确的下一步，最终张数由状态×背景合成决定；按 40% 配额需扩状态与合成背景。

### 垃圾贴纸 `waste_sticker`

现行 Train 1,800、Val 240、Test 240，共 2,280 张。每图 1–4 个可见桶，Train 已导出 4,050 个贴纸实例，Val、Test 各 540 个，完整图签审计通过。桶色保留场景变化和下游业务属性。分集配额和生成方式见[垃圾贴纸数据设计](DATASET_DESIGN.md#垃圾贴纸数据设计)。

| 来源 | 当前证据 | 已完成产量（张） |
| --- | --- | --- |
| `Waste Stickers` 场景直接出图 | 统一工程含四色桶和固定 24 类图案，已有受控合成诊断 | Train 990、Val 160、Test 200，按集合设置布局，基础资产共享 |
| Blender 空白桶面透视合成 | 24 张 Train 合格渲染底图、96 个表面 | Train 270，覆盖 24 张 Blender 桶面 |
| Blender Arena 通用空场 | 576 张，贴纸当前用于整帧空负 | 后续取得合成资格的成图仍计 Blender 派生，450／40 只计实拍场地混合 |
| Blender 整桶叠实拍场地 | 15 张合格实拍场地帧。Train 2 张分别来自 `full_scene.mp4`、`空场与场景物料_001.mp4`；Test 绑定 `junction_005` 13 张合格帧 | Train 450、Test 40，实际使用 1 张 Train 场地和 3 张 Test 场地，复用模板与布局逐图记录 |
| 实拍桶面透视合成 | 22 张合格，Train 的 `_001`、`_003` 各 1 张且有旧使用，Val 20 张均来自 `_004` | Train 90、Val 80，使用 2 张 Train 桶面和 20 张 Val 桶面 |
| P22 同源实拍 | 同组 32 张完整（视频 31 张及 P22 1 张）、11 张部分，002016 曾进 193 张历史候选包 | 另选四张同源回归，置于 2,280 张之外。暂无已确认的独立带贴实拍 Val／Test |

本批 Train 实际使用实拍底图合成 540 张，其余 1,260 张为无实拍底图的 Blender 派生，分别占 30% 与 70%。先前从桶状态用途筛出 88 张无桶场地候选，当时的桶状态归属为 Train 18、Val 19、Test 51。`junction_005` 的 13 张已复用地面和完整无贴标签取得整桶资格，并绑定贴纸 Test。`_004` 的 20 张桶面已绑定 Val。当前合格合成底图 61 个帧身份（Train 28、Val 20、Test 13），全量成图清单已记录 2,280 张和 5,130 个框，实拍与 Blender 的四路实际配额均已兑现。正式 Test 排除 `full_scene.mp4`，其他候选按实际缺项补判。已有合格的共享及 Blender 批次可直接推进，同一合格底图可复用到不同布局，各路实际用量按清单记录。44 张旧小样、193 张旧候选包及实测按历史回执追溯，旧文件已按授权清理。[逐图库](../datasets/annotated/common/sources/frame_environment.sqlite3) · [分配回执](../datasets/annotated/waste_bin/whole_bin_ground_reuse_20261117/production_split_assignment_receipt.json)

### 温湿度表 `meter`

train 3,500：12 个针位组合 × 291–292；反光约 875 张（25%）。

| 角色 | 父组 | 可用基数 | 当前已选/已用 | 状态与限制 |
| --- | --- | ---: | --- | --- |
| 直接实拍 | CarRealPicture / site_capture / 仪表视频 | 57 条视觉记录（45 帧 CarReal + 7 车载 + 2 归档 + 2 参考 + 1 任务点） | 0 | 无完整 8 点 Pose；11 条 direct_sample blocked、11 条 pose_reference candidate |
| 背景 | `full_scene.mp4` | 3 帧（3601、9901、10501） | 3（36 合成计划） | 同一父视频；3 帧均为 regression 复用 |
| 背景（候选） | 11 条 composite_background | 11 | 0（blocked） | 被排除原因是点位/标签未解，不是画面不可用 |
| 前景 | `station.blend` | 1 场景、12 针位状态 | 12 渲染状态（未产出） | 几何真值来自参数化针位；读数映射未确认 |
| Blender 背景 | arena master | 1 场景 | 12 渲染计划引用 | 与多个任务共享 |

结论：

- **直接实拍 350：缺口。** 无完整 8 点 Pose，当前真实帧不能作为 Pose 正例；需重建标注或维持排除。
- **实拍背景合成 1,400：受限。** 只有 3 帧、1 个父视频经计划采用；11 条被 blocked 的背景在解除点位/标签问题前不能扩池；1,400 张可由 12 前景 × 背景组合承载，但背景父组不足（<5）。
- **Blender 1,750（上限 1,750）：可行。** 站房与 arena 场景在册，12 个渲染状态是既定第一步，最终张数由前景×背景合成决定；参数化针位可扩全部 12 组合，按 50% 配额需扩状态与背景。

## 跨任务共享父组与合并账

| 父组 | 涉及任务 | 合并引用 | 说明 |
| --- | --- | --- | --- |
| `full_scene.mp4`（site_capture） | person、e_bike、sign（既有批次）、waste_bin_state、meter、building_name_paper、fire（楼名审核） | 7 个任务背景/直接候选 | 11,378 帧；同一父视频，跨任务派生不增父组 |
| `空场与场景物料_001.mp4` | person（281 背景）、e_bike（候选池 1,376 条引用） | 2 个任务 | 父链部分未核 |
| `场地多辆电动车_较低视角_003.mp4` | e_bike（直接 6 帧/审核）、sign（parking_003 3 帧）、共享池 | 3 个任务 | 同源复用 |
| `四色桶较低视角_004.mp4` | waste_sticker（背景角色）、共享池、person（5 帧审查，均 rejected） | 3 个任务 | 人物使用被几何判据拒绝 |
| CarRealPicture 批次 | waste_sticker（直接）、meter（45 帧）、fire（3 张预留背景）、building_name_paper（6 条记录）、person（3,289 条候选）、e_bike（575 张直接配额） | 6 个任务 | 1 视频 + 18 任务点图 + 2,773 帧；跨任务复用需按实际范围解释 |
| 楼体照片 / 楼宇视频 | fire、building_name_paper | 2 个任务 | 67 照片 + 8 视频 |
| `arena_master_user_20261113.blend` | person、e_bike、waste_bin_state、waste_sticker、meter | 历史母场景预算汇总包括人物、电动车、贴纸、仪表及桶状态 | 旧工程路径。现行按统一工程及各批实际清单记录谱系 |
| `arena_indoor_detailed.blend` | person（默认）、e_bike、sign/plate 部分入口 | 计划合计 ≥2,250 张（person 900、e_bike 600、sign 750） | 与 calibrated 有来源链；不是实测标定 |
| `three_buildings.blend` | fire、building_name_paper | 900 + 900 = 1,800 张 | 一个母场景 |
| `waste_bin_pbr.blend` 家族 | waste_bin_state、waste_sticker（照片标识/动画派生） | 1,400 + Blender 贴纸 | 同一父场景谱系 |

跨任务要点：

1. 同一实拍父视频供多个任务使用时，各任务分别记录用途和父组份额；结果按同源范围解释，不能声称独立采集或独立场次评估。
2. Blender 母场景按一个父组跨任务合并；同一场景的渲染图、机位和任务数不增加父组。当前没有跨任务父组账本（`docs/DATASET_DESIGN.md#blender-背景任务的父源复用` 已记为待实现），编译前需按共享场景汇总各任务配额。
3. `full_scene` 是 7 个任务的共同背景来源；若各任务都把它作为主要实拍背景，整个数据集的合成背景会集中在单一父视频上，与「不得集中在单一父视频」的要求冲突。

## 80% 父组参考与 val/test 预留

按任务和角色记录资格与分配。下表沿用历史快照的候选父组分母，80% 一列用于比较覆盖，末列保留当时的集中度判断。合格未分配现货按实际父组安排后纳入批次。

| 任务 | 角色 | 合格父组 | 已登记候选父组 | 80% 参考数（旧候选口径） | 当前 train 引用 | 已记录 val/test/回归预留 | 结论 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| person | 实拍背景 | 2–3 | 10 | 8 | 1（full_scene；001 候选未选） | 8 条 regression 记录（同父组） | 已审父组 <5，报告缺口 |
| person | 遮挡物 | 2 | 2 | — | 2 | 0 | <5，固定两姿态 |
| person | 身份图案 | 18（固定物料） | 18 | 不适用 | 2 | 0 | 固定 16+2 |
| e_bike | 实拍背景 | 1 | 3（来源计划分组） | 2 | 3 组已排份额 | 0 | train 用 3 组 >2，超出 |
| e_bike | 车辆前景 | 2 | 2（掩膜）；8（单车视频） | — | 2 | 0 | <5 |
| e_bike | 直接实拍 | 1（4 帧完整标签） | 5 审核视频 + 站点照片批次 | 4 | 3 组已排份额 | 7 张照片整组预留 | 合格 <5 |
| fire | 楼面背景 | 3（照片/楼体） | 67 照片 + 8 视频 | 53（按照片）/—（按楼体） | 3 楼面 | 5 背景 + 3 直接 + 1 图案预留 | 场次独立未证；按楼体 <5 |
| bnp | 纸条/背景 | 1（5 帧完整） | 3 楼面 + 2 视频父源 | — | 1（组合计划 2 父源） | 6 条 reserved 声明 | <5 |
| sign | 牌具 | 1 | 1（8 视图） | — | 1（6 train 掩膜） | 1 val + 1 test 掩膜（同父组） | <5，固定牌具 |
| sign | 背景 | 4 | 7（6 共享池视频 + 1 照片批次） | 5（按候选）/3（按已审） | 4 组（10 张） | 0 | 按已审 4 组算超出，需补审或确定口径 |
| waste_bin_state | 背景 | 1 | 7（含共享池） | 5 | 1 | 4 帧（同父组） | <5；预留非独立父组 |
| waste_bin_state | 桶体 | 4 | 8 视频 + 14 用户照片 | 6（按视频） | 0 直接 | 0 | 4 个固定实物 |
| waste_sticker | 带贴实拍 | 1（同源回归） | 1 | — | 旧候选包使用 002016 | P22 同组单列回归 | 现行直接实拍配额为 0 |
| waste_sticker | 桶面 | 旧 Train 两组红桶面；Val 已绑定一组 20 张合格帧 | — | — | 旧批 2 组 | Val 已按 004 整组分配 | 另有 Blender 96 个可贴正面 |
| waste_sticker | 实拍场地底图 | Train 2 张合格，Test 已绑定 `junction_005` 的 13 张 | — | — | Train 450 张为成图目标 | Test 40 张为成图目标 | 按地面几何和拍摄父组编排 split，按实际清单统计 |
| meter | 实拍背景 | 1 | 3（full_scene / CarReal / 车载） | — | 1 | 1 held-out | <5 |
| meter | 仪表模型 | 1 | 3（station/pbr/approx） | — | 1 | 0 | 固定模型 |

全量草案的 80% 按父组核算。批次按任务实际分集选取可用来源，已知共用父组的验证按同源范围报告。缺少独立父组时，其余已合格分支照常生成。

## 与全量草案对照：可行、受限、缺口

| 任务 | 直接实拍 | 实拍背景合成 | Blender | 主要依据 |
| --- | --- | --- | --- | --- |
| person | 缺口 | 受限 | 可行 | auto_validated 23（16 正、7 负）；human semantic 3；human box/absence 0；training admitted 0；pending partial 11；另有 200 个未穷举 candidate 父帧；已审背景 2–3 组；arena 在册 |
| e_bike | 缺口 | 受限 | 受限 | 4 帧完整标签；1/3 背景组合格；Blender 入口未批量 |
| fire | 缺口 | 受限 | 可行 | 直接全 blocked；多贴裁剪不支持；三楼合景可用 |
| building_name_paper | 缺口 | 受限 | 可行 | 5 帧完整；2 父源组合计划 40 张；Blender 1,200 配额、900 计划已写 |
| sign | 按完整图签使用 | 原有 585 张 Train，补证后另增 Train 70、Val 77 张合格底图 | 背景可复用 | 新增 14 个实拍父路径，场次独立性按证据记录 |
| waste_bin_state | 缺口 | 受限 | 可行 | 23 候选未审；背景 1 父视频；桶场景 19 状态 |
| waste_sticker | 当前批次 0 张 | 已导出 | 已导出 | 2,280 张、5,130 个框，按实际四路成图核算来源与同源范围 |
| meter | 缺口 | 受限 | 可行 | 无完整 Pose；3 帧 1 视频；站房参数化 |

解释：

- 「可行」只表示现有父源与入口计划能承载该分支配额，不表示画面质量、标签真值或模型效果已验证。
- 「受限」标明该分支的父组、标签、放置、入口或覆盖缺项。已有可行状态照常生产，同时补齐受影响的用途。
- 「缺口」记录该分支尚缺的可行容量。以可用来源编排当批、记录实际份额，按需补素材、补标注或调整状态。

## 父组安排与历史容量问题

1. **e_bike 历史背景份额集中。** 旧草案给 3 个实拍组安排 train 份额，当时只有 1 组合格，实际容量调给 G_full_scene。现行按已合格来源生产，80% 仅提示留源与评估覆盖。
2. **waste_sticker 的 Train 和 Val 按源视频分开。** 旧初筛曾将 6 个父视频的 18 帧标为 `train/ready`。现行编制从逐图库读取用途和实际 split，004 的 20 张合格桶面已整组绑定 Val。
3. **waste_bin_state 的 val/test 背景同源。** 4 个预留背景与 10 个 train 背景同属 `full_scene.mp4`，不构成独立父组预留；只能按同源复用解释。
4. **sign 的 val/test 掩膜同源。** 8 个掩膜全部来自 `禁止直行牌视频.mp4`；6/8 train 在掩膜层面是 75%，但父组层面只有 1 个。
5. **waste_sticker 旧批存在红色偏置。** 192 张旧实拍桶面合成均用红桶面。现行用自然变化的实拍与 Blender 桶面覆盖多种色彩，按有效实例和实际成像核对类别。
6. **直接实拍的现行设计目标合计 5,100 张。** 历史抽样按各层计数报告：person 为 available 34、annotated 23、candidate 34、auto_validated 23、human semantic 3、human box/absence 0、training admitted 0，pending partial 11。不能把 auto_validated 23 简写成“完整标签 23”或把 human semantic 3 改写成 human confirmation 3。CarReal/SiteCapture 的 200 个候选父帧因类别未知或未穷举仍保持 pending。当前可用容量按逐图库与实际计划核对。
7. **fire 多贴分支缺少受支持来源。** 现裁剪 `supported_counts: [0, 1]`；多贴 1,200 张需要新裁剪或 Blender。
8. **已修正：e_bike 空车组目标语义。** `upright_empty_street` 等 4 组的 `target` 已改为 `empty`，空车图不再带直立/倒伏目标类别；组配额未变。
9. **Blender 共享场景合并账。** 历史草案曾分别为 arena master、arena detailed 和三楼合景估算容量。现用统一工程，各批按实际 Scene、父资产和布局汇总。
10. **拍摄场次独立性未证。** 用户确认素材取自国赛真实场地、不同拍摄批次；具体场次、视频间独立关系未证实。跨任务和任务内的来源分组按保守假设记录，不能声称独立采集组。

## 需补标注、补渲染或补采的组

### 补标注（现有素材内，不需要新采集）

| 任务 | 需要补的内容 | 现有帧/图 | 保存位置 |
| --- | --- | --- | --- |
| person | 解决 pending partial；补足未穷举人物框与两街计数；继续确认空场；整理白背硬负例 | 80 条自动审查度量（46 张唯一帧）+ 34 个直接审查记录（auto_validated 23、pending partial 11）+ 200 个 candidate 父帧 + 281 帧背景候选 + `person_standees` 1,502 帧 | `datasets/annotated/person/`、`datasets/annotated/site_capture_20261112/person_vehicle_v4_review/`、`datasets/annotated/common/review/person/` |
| e_bike | 直接帧完整姿态/区域标签；背景街道/停车区标签；多车多实例 | 97 审核帧 + 160 车载图 + 2,773 帧 | `datasets/annotated/e_bike/` |
| fire | 整图火焰实例完整性；多贴框；负例空场 | 42 条用途投影 + 63 审核候选 + 67 楼体照片 | `datasets/annotated/fire/` |
| building_name_paper | 纸条框与逐字全文；多楼同屏归属 | 57 审核文件 + 5 完整记录 | `datasets/annotated/site_capture_20261112/fire_name_v4_review/`、`datasets/annotated/fire/` |
| sign | 完整牌具体框（含截断/杆）；无牌与普通立杆负例 | 8 掩膜视图 + 10 背景 | `datasets/annotated/sign/` |
| waste_bin_state | 逐桶开闭真值与桶掩膜 | 23 审核候选（21 帧） | `datasets/annotated/waste_bin_state/`、`datasets/annotated/waste_bin/` |
| waste_sticker | 用已有图签选同源回归帧，编制四路来源并核对每图完整的可见贴纸标签 | 现货见垃圾贴纸节 | `datasets/annotated/waste_bin/` |
| meter | 完整 8 点 Pose 与读数映射 | 57 条视觉记录 | `datasets/annotated/meter/` |

以上标注均在现有允许来源内进行；标注结果按任务保存，不能把候选观察当成完整真值。

### 补渲染 / 入口

- e_bike：Blender 批量计划入口与 `F_field_005_006` 前景接入；多车、人物遮挡、画外截断分支入口。
- sign：无牌具三维模型；除非先建牌具模型，否则 Blender 只能补背景，不能增加牌具父组。
- waste_bin_state：19 状态渲染后编译合成计划（现 `production_plan.json` 待掩膜生成）。
- meter：12 前景渲染后 36 合成计划；扩展反光与背景前需先解除 11 条 blocked 背景。
- waste_sticker：完整 Train／Val／Test 已导出并通过审计，训练使用当前数据入口。旧 26 状态和其他小样按持久回执追溯。
- fire、building_name_paper：三楼合景可渲染多贴/多名；需按共享场景合并计账。

### 补采

按现有证据，八任务 **train** 未发现「现有素材和合成都无法解决且实际阻塞任务」的必须采集项；sign、waste_sticker 的 val/test 独立场次补采为 `proposed`，meter、building_name_paper 为 `conditional`。这些是 split 独立性补采，不是 train 必须采集项。

- 固定物料（火焰图案、人物 16+2 身份、24 类贴纸、四色桶、牌具）是比赛现场实物，不因父组数少而补采。
- 直接实拍、桶面、火焰楼面、仪表背景等缺口均可由现有视频、照片或既有 Blender 场景经补标注/补渲染/扩背景池解决。
- 若未来 val/test 要覆盖「新场次」而非同源新画面，现有素材全部来自同一国赛场地，不能提供未见场次；这需要用户明确评估目的后再决定是否采集，且不属于本轮 train 目标。
- 当前素材中缺少 D 楼热红外数据与真值，属热异常任务，不在八任务数据范围，不能用可见光冒充。

## 未知项

- 视频间、照片批次间的拍摄场次独立关系未证实；用户只确认同场地、不同批次。
- person 281 帧背景的父链部分未核（审查计划中 21/42 帧 `parent_link_verified=true`）；其余帧的父视频关系未知。
- fire 67 张楼体照片的具体拍摄批次未逐个证实；63 个审核候选的场次关系 unknown。
- e_bike 街道/停车区位置标签未写入背景清单；50/50 位置配额无法按来源证明。
- sign 8 个掩膜的具体视面（正/侧/背）未在清单中标注；侧背视面覆盖无法按来源证明。
- Blender 场景的实际渲染容量、机位数量、渲染耗时未测量；注册表只证明资产状态，不证明可产出指定配额。
- 各任务 `production_plan.json` 与全量草案之间尚未编译；状态覆盖的容量缺口以 `python lab/scripts/contracts/state_coverage.py` 当前输出为准。

## 证据路径

- 任务来源配置：`datasets/scripts/<任务>/config/source.yaml`、`dataset.yaml`、各计划 JSON。
- [当前生产进度](DATASET_PRODUCTION.md#当前生产进度)与[全量 train 草案](FULL_TRAIN_PLAN_DESIGN.md#总览)。
- 训练目标与组合轴：`config/task_targets.yaml`、`config/combination_axes.yaml`；覆盖报告 `docs/DATASET_DESIGN.md#训练目标与组合轴覆盖生成`。
- 全量草案：`config/full_train_allocation.yaml`、`docs/FULL_TRAIN_PLAN_DESIGN.md`。
- 背景清单：`datasets/annotated/mixed_scene/v4_compositing_backgrounds_20261113/manifest.json`、`datasets/annotated/e_bike/v4_complex_backgrounds_20261112/backgrounds.json`、`datasets/annotated/waste_bin/v4_grounded_backgrounds_20261112/backgrounds.json`、`datasets/annotated/sign/v4_training_backgrounds_20261112/manifest.json`。
- e_bike 全量来源计划：`datasets/annotated/e_bike/v4_car_sequence_review_20261113/manifest.json`（`v4_combination_source_plan_20261114`）。
- 媒体查询库：`datasets/annotated/media_index/query_cache.sqlite3`（目录快照 2026-11-15T03:55:15Z）。
- Blender 登记库：`datasets/blender_source_registry.sqlite3`；查询 `python lab/scripts/blender/blender_source_registry.py list`。
- 原始来源映射：`datasets/source_copy_map.json`、`datasets/source_copy_manifest.jsonl`。
