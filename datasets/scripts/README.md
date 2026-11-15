# V4 数据集脚本入口

生产继续暂停。本轮只修改代码、配置和文档，执行无媒体写入的契约检查；没有制作小样、渲染、批量生产或训练。

每任务维护自己的 `config/source.yaml` 和 `config/dataset.yaml`。来源配置指向原始素材、现有标注清单、父来源和明确选定的记录，生成器实际读取自己支持的字段。候选发现清单不等于全部被选中。任务代码只导入本任务及 `common/`，共享素材可跨任务复用。

|任务|实际入口与已接通分支|消费的来源字段|当前选用范围和排除项|
|---|---|---|---|
|person|`compose_planned_ground_person.py` 接单人/多人、直立或倒伏车遮挡、已标注背景和明暗/饱和度；`annotate_real_person.py`、`render_person_arena_v4.py` 接实拍标注及 Blender；`inventory_auto_annotations.py` 盘点直接来源并输出分层状态|`person_artwork_root`、选定 `role_records` 的清单与记录位置、三种 `blender_*_scene`、`real_capture_root`|已填写 2 张正面人物纹理的头部/上身身份区域、先人后车的遮挡次序，保留身份区域至少 0.9 可见率；选 3 张完整标注背景。直接审查独立计数：source/available 34、annotated 23、candidate 34、auto_validated 23、human semantic 3、human box/absence 0、training admitted 0；另有 pending partial 11|
|e_bike|`compose_v4_grounded_train.py` 接零至多车、人物遮挡、允许的边缘截断及直接实拍；`render_e_bike_blender_batch.py` 与两个 Blender planner 接独立渲染分支|地面背景、车辆选择、人物图案、`direct_real_review_manifest`、`blender_scenes` 及设计/用途/前景清单|当前明确清单为 4 个背景、21 个候选车辆素材和 4 帧直接实拍；Blender 采用已登记用途前景，不自动把其他未登记素材冒充可用|
|fire|`render_blender_fire_dataset.py`；`compose_real_facades.py` 读取原照片裁剪并执行零贴/单贴、尺度和成像变化|`scene`、唯一 `target_artwork`、`real_facades` 的 3 张照片关系、裁剪及贴放区域|计划从 3 张原照片裁剪，不依赖旧小样；每照片零贴 2、单贴 4。完整原图监督和多贴分支不纳入当前计划|
|building_name_paper|`compose_rooftop_papers.py`、`plan_rooftop_batch.py`、`package_real_candidates.py`；`render_blender_names.py` 独立处理单楼、多楼及无纸条|字体、背景父来源、`rooftop_sources`、`combination_plan`、`direct_manifests`、场景与楼体集合|选 2 个完整检测正例、3 个完整负例，加 40 个原纸框内替字组合；原字串未知不做 OCR 监督，程序替字记录字符串。旧 1200 条全图标注完整度未知组合排除|
|sign|`compose_sign_v4_measured.py` 合成候选；`package_sign_candidate.py` 按计划集合打包|`foreground_root`、`foreground_video`、`background_manifest`、父来源范围及显式背景输入|选 10 背景、2 牌具视面；400 状态包含 32 个只保留下半部立杆/底座的设计裁切。直接实拍缺完整标签分支排除，不等待额外杆子实拍|
|waste_bin_state|`render_blender_waste_bin.py` → `plan_current_sources.py` → `generate_waste_v4_grouped.py`；另有场景渲染及导出分支|`background_manifest`、`standalone_scene`、`arena_scene`、`arena_bin_asset`、`asset_parent_scene`、`render_input`|已写 19 个渲染状态、选 10 背景并预留 4 张 val/test 背景。实际掩膜尚未生产；渲染后编译最多 360 张合成计划，不虚构当前不存在的掩膜|
|waste_sticker|`generate_sticker_real_photo_v4.py`、`render_sticker_arena_v4.py`；`package_real_stickers.py` 接指定直接实拍|P22、002011–002052 帧、002016–002044 停稳区间、`real_annotations`、父视频、24 图案清单、实拍几何、动态场景与来源计划|当前实拍只选 002016 的 4 类候选；40 个无标注源、P22 的 train 排除和 002044 的部分未知保持排除。合成选两个不同视频的桶面，24 类×4 变化＝192；另有 26 个 Blender 状态|
|meter|`render_meter_arena_v4.py` → `compose_meter_real_background_v4.py`；另有站房渲染入口|`station_source`、`station_face`、`endpoint_config`、`arena_source`、可选 `face_override`、实景背景记录及父来源|已写 12 个 RGBA/8 点前景渲染状态和 3 背景×12 前景的 36 张合成计划；完整 Pose 缺失的直接实拍排除。前景是恢复后的第一道工序，当前尚不存在|

仪表实景背景使用现有 review 的第 4、5、6 行，对应 `frame_003601`、`frame_009901`、`frame_010501`，保留原 regression 和候选负例状态。该清单的 `images` 行没有被当前媒体索引逐条解析，入口以 `review_record` 和原清单位置记录证据，清单作为 procedural 输入；媒体仍查询现有索引。没有另建数据库或伪造已索引关系。

`source.yaml` 的来源和设计权重是执行输入，来源数量不等于已生产张数，语义候选不等于人工验收。符合用途且标签可由几何推导的合成可按声明计划处理，打包保留实际标签状态，不新增逐图人工审批。直接实拍缺类、明确排除用途或标注范围不完整时按分支拒绝，其他分支可继续实现。

## 恢复生产入口

生产仍暂停，以下命令仅供恢复时执行。当前状态、首批预算、父来源范围及后续扩量预计由[生产记录](../../docs/DATASET_PRODUCTION.md#首批计划)统一维护；本节只列入口、计划和输出。

|任务|先运行的本任务入口|当前计划（本任务 `config/`）|输出|
|---|---|---|---|
|人物|`person/compose_planned_ground_person.py`|`production_plan.json`|`samples/person`|
|电动车|`e_bike/compose_v4_grounded_train.py`|`production_plan.json`|`samples/e_bike`；直接实拍为替代分支|
|火焰|`fire/compose_real_facades.py`|`production_plan.json`|`samples/fire`|
|楼名|`building_name_paper/compose_rooftop_papers.py`|`production_plan.json`|`exports/building_name_paper/train`；另有独立 Blender 计划|
|禁行牌|`sign/compose_sign_v4_measured.py`|`production_plan.json`|`samples/sign`，再按同一计划打包到 `exports/sign/train`|
|垃圾贴纸|`waste_sticker/generate_sticker_real_photo_v4.py`|`production_plan.json`|`exports/waste_sticker/train`；另有直接实拍及 Blender 计划|
|桶状态|`waste_bin_state/render_blender_waste_bin.py`|`render_plan.json`|`samples/waste_bin_state`；实际掩膜编译后写 `exports/waste_bin_state/train`|
|温湿度表|`meter/render_meter_arena_v4.py`|`render_plan.json` → `production_plan.json`|前景写 `samples/meter`；合成写 `exports/meter/train`|

以下命令在工作空间根目录执行。Blender 变量 `$DatasetBlender` 按[根目录说明](../../README.md#本机数据集入口)设置；使用已安装软件，GPU 入口采用 OPTIX。已有产物不能通过新建版本目录绕过输出限制；同一小样目录的替代分支不得连续写入非空目录。恢复生产前先运行 `python lab/scripts/contracts/workspace_layout.py check --task <任务>`；任务输出目录（`samples/<任务>`、`exports/<任务>/{train,val,test}`）尚未生成时只报提示，由生产守卫在生成时创建，不要预先创建空目录。训练目标与组合轴覆盖、全量容量缺口见[训练目标与组合轴覆盖](../../docs/DATASET_DESIGN.md#训练目标与组合轴覆盖生成)；容量缺口不为零时不能只放大预算。全量计划设计草案见[全量训练计划设计](../../docs/FULL_TRAIN_PLAN_DESIGN.md)；草案是 PlanDraft，不覆盖当前 `production_plan.json`。

```powershell
# 人物：重算直接来源分层状态；pending 不会自动改成 rejected/excluded
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/person/inventory_auto_annotations.py

# 人物：已选身份区域及直立/倒伏车遮挡
.venv/Scripts/python.exe -B datasets/scripts/person/compose_planned_ground_person.py --production-plan datasets/scripts/person/config/production_plan.json --output datasets/samples/person

# 人物：全量三批（train 3000 / val 300 / test 300）；输出目录必须不存在，或与本批已落回执完全一致
.venv/Scripts/python.exe -B datasets/scripts/person/compose_planned_ground_person.py --production-plan datasets/scripts/person/config/full_train_plan_3000.json --output datasets/exports/person/train
.venv/Scripts/python.exe -B datasets/scripts/person/compose_planned_ground_person.py --production-plan datasets/scripts/person/config/val_plan_300.json --output datasets/exports/person/val
.venv/Scripts/python.exe -B datasets/scripts/person/compose_planned_ground_person.py --production-plan datasets/scripts/person/config/test_plan_300.json --output datasets/exports/person/test

# 电动车：合成主分支；直接实拍入口是替代分支，不连续写同一非空目录
.venv/Scripts/python.exe -B datasets/scripts/e_bike/compose_v4_grounded_train.py --plan datasets/scripts/e_bike/config/production_plan.json --output datasets/samples/e_bike
# 替代：.venv/Scripts/python.exe -B datasets/scripts/e_bike/prepare_v4_low_angle_real.py --plan datasets/scripts/e_bike/config/direct_real_plan.json --output datasets/samples/e_bike

# 火焰：原照片裁剪区域内的零贴/单贴
.venv/Scripts/python.exe -B datasets/scripts/fire/compose_real_facades.py --plan datasets/scripts/fire/config/production_plan.json --output datasets/samples/fire

# 楼名：完整检测实拍与原纸框内替字
.venv/Scripts/python.exe -B datasets/scripts/building_name_paper/compose_rooftop_papers.py --plan datasets/scripts/building_name_paper/config/production_plan.json --output datasets/exports/building_name_paper/train

# 禁行牌：生成当前小样，局部索引，再打包
.venv/Scripts/python.exe -B datasets/scripts/sign/compose_sign_v4_measured.py --output datasets/samples/sign --seed 1918 --plan datasets/scripts/sign/config/production_plan.json
.venv/Scripts/python.exe -B lab/scripts/media/media_index.py build --scan-root datasets/samples --source datasets/samples/sign --no-video-probe --no-clean
.venv/Scripts/python.exe -B lab/scripts/media/media_index_sqlite.py sync --partition datasets/samples/sign
.venv/Scripts/python.exe -B datasets/scripts/sign/package_sign_candidate.py --source datasets/samples/sign --plan datasets/scripts/sign/config/production_plan.json --output datasets/exports/sign/train

# 垃圾贴纸：两个实拍桶面的24类组合
.venv/Scripts/python.exe -B datasets/scripts/waste_sticker/generate_sticker_real_photo_v4.py --plan datasets/scripts/waste_sticker/config/production_plan.json --out datasets/exports/waste_sticker/train --seed 20261114
```

桶状态的渲染图与掩膜尚不存在，`production_plan.json` 因此尚未生成。这是已实现的工序依赖，不再等待选源或补实拍。渲染得到实际可见记录后，规划器一次性选用已定 10 背景和两种色调，排除空前景及不适用放置，写回本任务固定计划；360 是上限，实际张数由返回记录决定。

```powershell
& $DatasetBlender --background datasets/assets/blender/waste_bin/blender_pbr_v2_20261110/waste_bin_pbr.blend --python-exit-code 1 --python datasets/scripts/waste_bin_state/render_blender_waste_bin.py -- --out datasets/samples/waste_bin_state --plan datasets/scripts/waste_bin_state/config/render_plan.json --seed 1918 --samples 12 --device OPTIX
.venv/Scripts/python.exe -B lab/scripts/media/media_index.py build --scan-root datasets/samples --source datasets/samples/waste_bin_state --no-video-probe --no-clean
.venv/Scripts/python.exe -B lab/scripts/media/media_index_sqlite.py sync --partition datasets/samples/waste_bin_state
.venv/Scripts/python.exe -B datasets/scripts/waste_bin_state/plan_current_sources.py --seed 1918
.venv/Scripts/python.exe -B datasets/scripts/waste_bin_state/generate_waste_v4_grouped.py --out datasets/exports/waste_bin_state/train --plan datasets/scripts/waste_bin_state/config/production_plan.json
```

仪表的场景、站房模型、表面范围及端点配置已接入。先按 12 状态渲染 RGBA 与 8 点，再刷新该分区，让后续守卫读取实际来源。渲染输入和合成输出使用不同目录；完整 Pose 缺失的直接实拍不进入此批。

```powershell
& $DatasetBlender --background datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend --python-exit-code 1 --python datasets/scripts/meter/render_meter_arena_v4.py -- --output datasets/samples/meter --production-plan datasets/scripts/meter/config/render_plan.json --count 12 --pose --foreground --device OPTIX --samples 16
.venv/Scripts/python.exe -B lab/scripts/media/media_index.py build --scan-root datasets/samples --source datasets/samples/meter --no-video-probe --no-clean
.venv/Scripts/python.exe -B lab/scripts/media/media_index_sqlite.py sync --partition datasets/samples/meter
.venv/Scripts/python.exe -B datasets/scripts/meter/compose_meter_real_background_v4.py --production-plan datasets/scripts/meter/config/production_plan.json --count 36 --output datasets/exports/meter/train
```

已经写好的独立分支如下。垃圾贴纸 002016 仍保留 4 类候选标注状态；其 Blender test 使用共享图案，不能作为独立实采评估。楼名 900 张 Blender 计划独立于首批 45 张，不声称已经渲染或合并。

```powershell
.venv/Scripts/python.exe -B datasets/scripts/waste_sticker/package_real_stickers.py --plan datasets/scripts/waste_sticker/config/direct_real_plan.json --out datasets/samples/waste_sticker
& $DatasetBlender --background datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend --python-exit-code 1 --python datasets/scripts/waste_sticker/render_sticker_arena_v4.py -- --plan datasets/scripts/waste_sticker/config/blender_plan.json --out datasets/exports/waste_sticker/test --seed 20261114 --samples 4 --max-seconds 360
& $DatasetBlender --background datasets/assets/blender/fire/blender_buildings_v13/three_buildings.blend --python-exit-code 1 --python datasets/scripts/building_name_paper/render_blender_names.py -- --plan datasets/scripts/building_name_paper/config/blender_plan.json --output datasets/samples/building_name_paper --device OPTIX
```

## 从现有数据库取得当前输入

`lab/scripts/media/media_index_sqlite.py task-inputs` 查询现有 `query_cache.sqlite3` 中的任务声明。数据库保存八份来源/设计配置及明确引用的 `production_plan`、`render_plan`、`direct_real_plan`、`blender_plan`；生成器继续读取返回路径所指的本任务 YAML/JSON。数据库用于发现和反查，原配置仍是编辑源，不建立第二套计划。

```powershell
# 所有任务的当前计划、预算、集合、存在状态与路径
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index_sqlite.py task-inputs --limit 0
# 按任务、集合和计划字段角色取得输入引用
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index_sqlite.py task-inputs --task person --kind input --split train --role background --limit 0
# 当前来源配置，含选定记录、排除原因及父来源；桶的待合成背景ID也在这里
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index_sqlite.py task-inputs --task waste_bin_state --kind config
# 读取数据库里的精确状态声明；按返回的plan_path交给既有生成入口
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index_sqlite.py task-inputs --task meter --kind state --plan-key render_plan --limit 0
# 从一张图反查哪些任务计划声明使用它
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index_sqlite.py task-inputs --kind input --path datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_001201.png --limit 0
```

`input` 的角色按状态字段定位为 `background`、`target`、`parent_source`、`annotation_reference` 或 `declared_input`，并保留 `reference_fields` 和状态 ID；它不是视觉适用性的新判定。未展开到状态的用途及排除项通过 `config` 读取，不能把整个候选池当作已选输入。`present=false` 表示文件当前不存在；桶状态合成计划为 `pending_plan`，仪表未来前景仍是计划引用。状态原文保留候选标签和来源说明。

查询会核对声明文件的存在状态、大小和修改时间；不一致时拒绝返回过期声明，并指出需刷新的任务分区。编辑某任务 YAML/计划后，仅构建该分区并同步。示例：

```powershell
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index.py build --scan-root datasets/scripts --source datasets/scripts/person --no-video-probe --no-clean
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index_sqlite.py sync --partition datasets/scripts/person --partition datasets/scripts/@root
```

构建器同时维护扫描根的 `@root` 元数据；未变化的分区复用原快照。桶状态在实际渲染后运行 `plan_current_sources.py` 创建计划时，也须按上述方式刷新 `datasets/scripts/waste_bin_state` 分区。媒体新增仍按各自 `samples` 分区刷新，配置声明刷新不代替媒体索引。所有这些索引操作都不执行生产。

源目录移动或失效会改变其他分区的 `source_present`；此时只按上面的 scoped `sync` 会留下 `complete_catalog=false` 的部分缓存，`task-inputs` 随即以 partial cache 拒绝。须补一次不带 `--partition` 的全量 `sync`。配置或计划声明变化时还须先重建 `datasets/scripts` 声明分区，再执行全量 `sync`。2026-11-15 搬走 `datasets/person/` 后实测：scoped sync 后为 215 分区、`complete_catalog=false`，全量 sync 恢复 220，再接入 `datasets/assets/person` 分区后为 222；随后重建声明分区并同步得到 223 分区、`complete_catalog=true`。

任务 YAML 里的日期等标量要加引号（`'2026-11-15'`）。未加引号时 PyYAML 会解析成 `date` 对象，构建索引写 JSON 时因不可序列化而失败（`TypeError: Object of type date is not JSON serializable`）；这是当前构建器的已知限制，尚未修改代码。

## 背景审查入口

`datasets/scripts/common/review/` 提供跨任务候选帧审查：采样与近重复、地面几何、目标缺失扫描、掩膜质量、判定与证据写入。任务级计划在 `datasets/scripts/<任务>/config/review_plan.json`；判定写入 `datasets/scripts/<任务>/config/review_selection.json`（入 Git；人物任务的 B6 结果已由 `compile_background_dispatch.py` 接入当前生成计划）；度量、叠加图与批次清单写入 `datasets/annotated/common/review/<任务>/<批次>/`，透视模板由 `perspective` 写入 `datasets/annotated/common/perspective/<任务>/<批次>.json`（均忽略，可由代码和原图重算）。审查产物、判据和父源筛查属于跨任务 common，人物只是第一个使用者。

```powershell
# 1) 生成待审清单（只规划，不读图）
.venv/Scripts/python.exe -X utf8 -B -m datasets.scripts.common.review.cli plan --task person
# 2) 跑度量并按 ready/unknown 写叠加图
.venv/Scripts/python.exe -X utf8 -B -m datasets.scripts.common.review.cli analyze --task person --overlay
# 3) 写判定（先看统计再落盘）
.venv/Scripts/python.exe -X utf8 -B -m datasets.scripts.common.review.cli apply --task person --dry-run
.venv/Scripts/python.exe -X utf8 -B -m datasets.scripts.common.review.cli apply --task person
# 4) 覆盖统计
.venv/Scripts/python.exe -X utf8 -B -m datasets.scripts.common.review.cli report --task person
# 5) 导出可复用的地面/地平线透视模板
.venv/Scripts/python.exe -X utf8 -B -m datasets.scripts.common.review.cli perspective --task person
```

判定语义：`ready`／`excluded`／`unknown`，`unknown` 是一等结果。`templates` 为空时目标缺失扫描永不判干净（`emptiness_unproven`）；地面估计失败记 `unknown` 而非 `excluded`，因为估不出地面不等于该帧不可用；地面边缘支持度只作为限定条件记录，不作为门禁。判定先由脚本计算；人物 B6 的 11 条确定判定已由用户接受，8 条 unknown 保持 pending。记录保留判定理由和接受范围；其他任务的自动判定不因此自动获得人工确认。

`apply` 保留同批次、同帧、同判定的人工接受字段；切换批次时旧记录放入 `previous_batches`。判定只适用于相应批次和帧。判定由脚本判据自动给出，视觉复核方法与人工验收口径尚未确定；在当前规则下，计划里由人工给出的 `placement_patch` 不因自动几何判 `unknown` 而失效，但也不能算作已通过自动审查。`perspective` 只导出有地面的帧，`unknown` 不被提升为透视模板。

地面净空门禁 `min_patch_free_ratio`、`max_patch_obstacle_components` 仍作为兼容预筛，默认值 0 表示不启用。放置区域杂物门禁由 `review/placement.py` 实现：`patch_generated` 几何阶段之后检查 patch 内对象占用、最大对象组件、垂直结构、最大自由矩形、可用脚点比例和最佳脚点身体框；`placement.required` 未写时保持旧行为。`placement_patch_cluttered` 判 `excluded`，`placement_patch_unsure` 判 `unknown`；薄长地面标线单独记录 `marking_ratio`，与本任务已有目标框重叠的组件记为 `task_target`，都不算杂物。B6 的人物任务阈值由用户接受的 24 帧分类拟合，24/24 是标定集合内一致数。当前人物 `review_plan.json` 和 `review_selection.json` 均为 B6，结果为 6 ready／8 unknown／5 excluded；8 条 unknown 保持 pending。6 条 ready 通过 `background_dispatch.json` 分配给人物 train 的现有 54 个候选状态，预算 36。公共审查模块仍服务所有任务，阈值和目标语义由各任务提供。

地面几何在 VP patch 失败时按顺序回退：先尝试水平相机 patch（`placement_patch_level_camera`），再尝试 Gabor/FFT 纹理地面顶边（`horizon_texture_fallback`、`placement_patch_texture_fallback`），并要求回退 patch 达到最小深度，避免把只剩窄地面的帧硬救回来。`texture.py` 提供 Gabor 滤波器组与一维纹理变点搜索；回退只在主路径失败时运行，reason code 保留使用过的路径。

审查入口不在 `lab/scripts/audit/test_production_entry_coverage.py` 的扫描范围（该工具跳过 `datasets/scripts/common/`），其写入不受静态门控；它只写 `annotated/common/` 与任务 `config/`，不写 `samples/`、`exports/`。

## 背景与来源总账

`datasets/scripts/common/source_ledger/` 从各任务 `config/source.yaml` 的 `pairing`（存在时）、批次清单、共享背景池 manifest 和 Blender 登记库重算跨任务背景/来源总账，报告写入 `datasets/annotated/common/sources/ledger.json`。它是派生视图，不是新的来源权威；没有 `pairing` 的任务只记录 legacy 观察并报告 adoption，不按目录名推断父组或 split。配对准入以 `python lab/scripts/contracts/source_pairing.py --check` 为单一 checker；source_ledger 的 `check`/`write` 是派生报告诊断，不是第二套 gate。

```powershell
.venv/Scripts/python.exe -m datasets.scripts.common.source_ledger.cli check
.venv/Scripts/python.exe -m datasets.scripts.common.source_ledger.cli write
.venv/Scripts/python.exe -m datasets.scripts.common.source_ledger.cli query --task person --role composite_background --split train
.venv/Scripts/python.exe -m datasets.scripts.common.source_ledger.cli matrix
```

检查码、背景可用性规则、输出字段和能力边界见[背景与来源总账](../../docs/BACKGROUND_SOURCE_LEDGER.md)。

## 公共能力和输出

`common/bayesian.py` 复用现有分配器及生产规划器，`perspective.py`、`images.py`、`composite.py`、`annotations.py`、`paper.py`、`blender_standees.py`、`blender_masks.py` 提供共享变换、成像和标签操作，`placement.py` 提供放置区域杂物评估，`texture.py` 提供几何回退用的 Gabor/FFT 纹理边界。Blender 缺 PyYAML/SciPy 时沿用工作空间解释器回退，不加载外部脚本。任务目标、类别、来源选择和布局留在任务目录。

人物计划支持可选状态及小于候选状态数的预算；楼名 Blender 外部计划支持弹性 `min_count/max_count/budget`，实际按 `planned_count` 执行。默认固定覆盖计划仍按固定配额执行，不能将固定配额描述为先验改变了分配。

每任务只有一个 `datasets/samples/<task>`，正式数据只在 `datasets/exports/<task>/{train,test,val}`。当前八任务的上述目录均无生产文件。只有声明支持对应集合的入口接受正式输出，候选专用分支仍输出小样。生成入口在守卫内写出，声明自身两个 YAML、实际输入、代码和输出。旧入口转发到当前实现，已退休公开写入口直接拒绝；纯计划只打印 JSON 或写本任务已有配置计划区。V3 模型、素材、历史回执及媒体数据库未因本次代码修复发生变更。

## 离线核对

`lab/scripts/audit/test_production_entry_coverage.py` 不导入或执行生成器，区分 `DISABLED`、`FORWARDED`、`PLAN_ONLY`、`GUARDED`、`NO_STATIC_WRITES`、已核对条件分支，以及使检查失败的 `BYPASS`、`UNPROVED`、未分类项。它解析兼容转发的真实目标，检查公开生产函数及 `main`/`package`，不会因为文件里出现“retired”字样就忽略可达写入。

自测命令是 `python -B lab/scripts/audit/test_production_entry_coverage.py --self-test`，整体盘点命令去掉 `--self-test`。公共配置、固定输出和禁止跨任务导入的契约位于 `common/tests/test_contracts.py`，各任务新增测试验证实际字段和调用关系。检查结果只覆盖代码契约和词法路径；渲染效果、像素可读性及模型效果尚未执行。

2026-11-15 复核：`lab/scripts/audit/test_production_entry_coverage.py --self-test` 通过 15 项审计自测；静态盘点为 227 个文件、402 个入口记录，未受管写入、未证明转发和未分类项均为 0。配置与代码契约测试以 `common/tests` 和各任务 `test_*.py` 的当前输出为准，本轮未重跑完整套件；当前可复现的计数是上述 15 项自测和静态盘点记录。检查不执行 RGB 合成、渲染或训练；词法审计不证明动态路径或实际渲染结果。
