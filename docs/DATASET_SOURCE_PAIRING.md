# 数据集父子源配对与分集预留设计

> 2026-11-15 配对设计。逐任务矩阵和补采清单记录初始预留提议。当时只给 train 定预算，val/test 先预留父组。实际选源、计划绑定和冲突看各任务当前 `source.yaml`。已执行批次见[当前生产进度](DATASET_PRODUCTION.md#当前生产进度)。
>
> 编号命名空间、任务口径和计数单位见[编号与计数口径](ID_AND_COUNT_POLICY.md)；现有 R/V/BSL 编号冻结，新规则使用 `PAIR-RNN` / `PAIR-VNN`。

## 1. 需求提取

| ID | 需求 | 来源 |
| --- | --- | --- |
| R1 | 每个任务的 raw 有两个来源类别：本任务实拍、背景；父源只按这两类定义 | 用户 2026-11-15 |
| R2 | 本任务实拍素材要匀到 train/val/test，再在各 split 内用组合轴排列组合合成更多图 | 用户 2026-11-15 |
| R3 | 本任务外其他 raw 用审查工具筛查为背景，与数据源合成 | 用户 2026-11-15 |
| R4 | 合成示例：person 用 `team_assets` 数字图 + 背景；fire 用火焰数据源 + 楼宇/其他背景；building_name_paper 用 Blender 楼体 + 楼；e_bike 用平视旋转数据源 + 其他背景 | 用户 2026-11-15 |
| R5 | 通用 Blender 母场景也是合成来源之一 | 用户 2026-11-15 |
| R6 | 所有合成都经贝叶斯分配额度、组合状态 | 用户 2026-11-15 |
| R7 | 多数据集可复用同一批素材；因分开训练，复制完全相同的图片可接受，前提是各任务学习的特征不同 | 用户 2026-11-15 |
| R8 | 配对结构放在各任务 `source.yaml` | 用户 2026-11-15 |
| R9 | 单父组任务补采独立场次 | 用户 2026-11-15 |
| R10 | 扩用共享背景池 | 用户 2026-11-15 |
| R11 | val/test 先只预留父组，不定张数 | 用户 2026-11-15 |
| R12 | 结构需覆盖大量变量，结构化、矩阵化 | 用户 2026-11-15 |

项目既有约束继续适用：不按帧数冒充独立来源；派生数据继承父来源；同一任务内不得把逐字节完全重复的样本分到不同监督集合；跨任务复用需记录实际范围；不提交数据、权重或凭据。

## 2. 变量空间

### 2.1 三层概念必须分开

| 层 | 回答的问题 | 本设计中的角色 |
| --- | --- | --- |
| 父源类别 | 这是本任务实拍，还是背景？ | 帮助安排 split 并解释 val/test 的来源重叠；不决定素材能否开始制作 |
| 组合轴状态 | 在同一 split 内要覆盖哪些视角、光照、颜色、楼面等？ | 决定合成状态和覆盖矩阵，不决定 split |
| 数据源角色 | 合成时它是目标、遮挡物还是背景？ | 决定标签和图层，不单独算父组 |

一个容易混淆的点：**遮罩、裁片、掩膜是数据源，不是新的父组**。它们保留原父组身份。跨 split 使用时记录同源复用，所得 test 用于同源诊断；需要独立来源评估时另选未接触父组。

父源类别随画面内容和任务用途登记。一个父视频可以为任务 A 提供完整实拍监督，为任务 B 提供可合成背景。`raw/<任务>/targets/` 可保存该任务的完整相机负例，已证实全帧无任务目标即按 `empty_negative` 查询。其他任务目录中的同帧也可凭本任务完整标签使用；当前资格与来源关系在逐图库查询。[原始来源定位](../datasets/README.md#原始来源定位)

同一父视频可同时提供任务实拍帧和背景帧，例如 `full_scene` 的人物段与空场段。父组登记一个主 split，按计划复用到其他 split 时标明同源。针对父视频的组外评估按整段留出。

### 2.2 维度清单

| 维度 | 取值/说明 |
| --- | --- |
| 任务 | person / e_bike / fire / building_name_paper / sign / waste_bin_state / waste_sticker / meter |
| 父源类别 | `task_real`（画面含本任务目标的真实采集，可在本任务目录或共享采集域）、`background`（画面不含本任务目标、用作场景/遮挡/空场的采集，含其他任务 raw、归档、共享池、Blender 背景） |
| 父组单位 | `parent_video`、`parent_photo`、`parent_photo_batch`、`physical_object`、`field_session`、`blender_scene` |
| 数据源类型 | `fixed_artwork`（数字图案/身份图）、`real_cutout`（实拍裁片）、`mask`（掩膜）、`blender_target`（Blender 目标模型/场景）、`generated_text`（生成文字） |
| 合成角色 | `direct_target`、`composite_target`、`occluder`、`composite_background`、`empty_negative`、`geometry_reference`、`material_reference`、`mask_source`、`target_foreground`、`ocr_target`、`ordinary_pole`、`reference`、`surface` |
| 使用方式 | 直接使用、合成使用、纯渲染、负例 |
| split | `train`、`val`、`test`、`shared`（固定物料/Blender 场景）、`inherit`（派生数据源继承父组）、`unassigned`、`to_review` |
| 组合轴 | 按任务取 `config/combination_axes.yaml` 的轴与值；`dynamic` 轴由来源审查状态填充 |
| 标签状态 | `complete`、`partial`、`unknown`、`blocked`、`excluded`、`candidate`、`ready` |
| 背景审查状态 | `confirmed_empty`、`candidate_empty`、`contains_target`、`unknown`、`rejected` |
| 采集场次 | `proven`、`unknown`、`same_session` |
| 复用 | 跨任务、跨 split（同源）、派生自某父组 |
| 证据 | manifest、record_index、JSON pointer、review 输出、计划路径 |
| 数量 | 帧/图/实例/片段；只作基数，不作独立父组数 |
| 计划绑定 | train、val、test 各自的执行计划路径。未编制的 split 保留空值 |
| 采集动作 | `selected`、`reserved`、`to_review`、`to_label`、`to_extract`、`to_capture`、`blocked` |

### 2.3 来源 × 用途矩阵

| | 直接使用 | 合成使用 |
| --- | --- | --- |
| **任务实拍** | 直接实拍图分到各 split | 任务实拍前景/掩膜 + 背景合成 |
| **背景** | 确认空场/负例图（如 300 张空街） | 数据源（图案/数字图/Blender 目标）+ 背景合成 |
| **Blender 母场景** | 纯渲染背景或负例 | 背景层，或目标层（楼宇、站房、桶、贴纸场景） |

**开始分配 split 的最小输入**是任务、待用素材的帧身份或可定位路径、已知父来源，以及本批希望生成的 train/val/test 用途。父来源暂不确定时先登记 `unassigned` 和待核关系，仍可规划其他已知来源。对已有用途资格的未分配现货可直接编排批次，并按已知父来源选定 split。正式生成按本次所用帧核对目标标签、用途和放置空间。先记录各父组拟用于哪些集合，再按现有可行状态和成图预算编排；同一合格原帧可生成不同布局，来源类型比例见[来源组成](#410-来源组成)。

**分配后的检查**按目的分层：训练批次核对选中图像的身份、标签、用途和图签；val 可用于同源调参，但要标明与 train 共享的父视频、物料或 Blender 母场景；只有声称独立来源 test 时才整组留出未接触父源。逐字节相同的最终样本不跨监督集合。来源数、80% 参考份额和未完成候选只提示覆盖与评估范围，不推迟已可行的制作。

## 3. `source.yaml` 配对结构

### 3.1 放置与权威

- 每个任务的 `config/source.yaml` 在 `pairing` 中维护父子来源、主 split、用途预留和证据路径。
- `current_selection` 与 `production_plan` 指向当前选择及执行计划。批次清单保存逐图用途，通过 `manifest` 或 `record_index` 关联父组。
- 批次回执绑定入口声明的配置与实际输入。编制新计划时读取对应任务的 `source.yaml`。
- 一个父组在一个任务内只能有一个主 split；同源跨 split 用 `same_source_allowed_in` 明确列出，不能默认。

### 3.2 结构示例

```yaml
pairing:
  schema_version: 1
  status: draft                 # draft | reserved | executable
  composition_policy:
    target_source: video_frame_masks
    blender_priority: background_only
    note: 目标数据源为牌具视频帧裁片；Blender 只作背景
  policy:
    parent_classes: [task_real, background]
    split_unit: parent_group
    train_parent_cap: 0.8
    val_test_mode: reserve_only # 先只预留父组，不写张数
    derived_inherits_parent_split: true
    cross_task_duplicates: allowed_with_separate_labels
    same_source_splits: record_and_report
  task_real:
    - id: tr_person_standees
      class: task_real
      kind: parent_video
      paths:
        - Archive/Assets/captures/site_capture_20261112/videos/person_standees.mp4
      evidence:
        - {manifest: datasets/annotated/person/v4_multivideo_review_20261112/review_manifest.json, record_index: 0}
      capture_session: unknown
      labels: {status: partial, note: 8 帧 AI 候选，未人工验收}
      roles: [direct_target, composite_target]
      axes: {view: [front, oblique], lighting: [normal, adjusted]}
      split: {primary: train, status: proposed}
      plan: null
  backgrounds:
    - id: bg_full_scene
      class: background
      kind: parent_video
      paths:
        - Archive/Assets/captures/site_capture_20261112/videos/full_scene.mp4
      evidence:
        - {manifest: datasets/annotated/mixed_scene/v4_compositing_backgrounds_20261113/manifest.json, record_index: 0}
      review: {state: candidate_empty, method: contact_sheet, accepted_by: null}
      labels: {status: unknown}
      roles: [composite_background, empty_negative]
      split: {primary: train, status: proposed, same_source_allowed_in: [val]}
      plan: null
  data_sources:
    - id: ds_identity_community_1
      kind: fixed_artwork
      path: Archive/Assets/working_sources/digital/person/team_assets/社区人员/1.png
      parent_group: null
      roles: [target_foreground]
      split: {primary: shared, status: confirmed}
  blender:
    - id: bl_arena_master
      kind: blender_scene
      path: datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend
      role: background_scene
      split: {primary: shared, status: confirmed, same_source_note: 多任务共享母场景}
  captures:
    - id: cap_sign_val_20261116
      target_split: val
      reason: no_independent_parent
      status: proposed
      object: 禁止直行牌具
      scene: 与现有场次不同的路口或白墙摆位
      views: [front, oblique, side, back]
      lighting: [normal, dark, bright]
      amount: 2–3 分钟视频或 100–200 张照片
      truth: [fixture_box, view, truncation, placement]
      save_to: Archive/Assets/captures/sign/capture_20261116_val/
  plan_bindings:
    train: [datasets/scripts/sign/config/production_plan.json]
    val: []
    test: []
```

### 3.3 字段说明

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `composition_policy` | 建议 | 任务级目标数据源与 Blender 优先级；fire/bnp/waste_bin/waste_sticker/meter 用 `blender_primary` + `increased`，person/e_bike/sign 用 `background_only`；不写张数 |
| `composition_policy.target_components` | 建议 | 目标侧的机读组件，例如 `team_assets_digital_person`、`fixed_flame_artwork`、`generated_text`、`bin_model`、`animated_sticker_scene`、`station_model` |
| `composition_policy.background_components` | 建议 | 背景侧组件，例如 `arena_scene`、`three_buildings_scene`、`real_backgrounds` |
| `composition_policy.blender_share_target` | 建议 | 提高 Blender 的任务写明目标份额；`blender_priority: increased` 时使用 |
| `id` | 是 | 任务内唯一；建议前缀 `tr_`（任务实拍）、`bg_`（背景）、`ds_`（数据源）、`bl_`（Blender）、`cap_`（采集） |
| `class` | 是 | `task_real` 或 `background`；数据源和 Blender 场景另在各自列表 |
| `kind` | 是 | 父组单位，见 2.2 |
| `paths` | 是 | 工作空间相对路径；一个父组可有多个文件或批次 |
| `evidence` | 是 | manifest + record_index / JSON pointer / review 输出 |
| `capture_session` | 任务实拍必填 | `planned` / `proven` / `unknown` / `same_session`；`planned` 表示采集请求已登记但尚未执行，采后改 `proven` |
| `capture_id` | 计划补采必填 | 指向 `captures` 的请求 ID；占位父组与采集请求不得只靠路径隐式关联 |
| `labels.status` | 是 | `complete` / `partial` / `unknown` / `blocked` / `excluded` |
| `review.state` | 背景必填 | `confirmed_empty` / `candidate_empty` / `contains_target` / `unknown` / `rejected` |
| `roles` | 是 | 合成角色列表；同一父组可有多个角色，但同一任务内角色与 split 要一致 |
| `axes` | 建议 | 该父组已能证明的轴值；`dynamic` 轴留待审查填充 |
| `split.primary` | 是 | `train` / `val` / `test` / `shared` / `unassigned` |
| `split.status` | 是 | `proposed` / `reserved` / `confirmed` |
| `split.action` | 建议 | 采集动作：`selected` / `reserved` / `to_review` / `to_label` / `to_extract` / `to_capture` / `blocked`；计划补采占位用 `to_capture` |
| `split.same_source_allowed_in` | 可选 | 记录当前计划允许共享的 split，组外评估按该共享维度解释 |
| `parent_group` | 派生数据源必填 | 指向 `task_real` 或 `background` 的父组 ID；固定图案/生成文字为 `null` |
| `plan_bindings` | 可选 | 当前 split 关联的计划路径 |
| `plan_alignment` | 可选 | 现有计划与 pairing split 的对齐状态：`status: aligned/stale/unknown`、`note` 和 `conflicts`；`stale` 表示现有计划仍消费其他 split 的父组，编译前必须重编或改 split |
| `plan_alignment.conflicts` | 可选 | 逐条记录 `parent`、`plan`、`pairing_split`、`plan_use`；只作待决标记，不修改计划 |
| `captures` | 可选 | 独立场次补采请求；字段按 6 节 |
| `captures.status` | 是 | `proposed` / `conditional` / `confirmed`；与 `task_real` 占位的 `capture_id` 关联 |

### 3.4 校核规则

| ID | 规则 |
| --- | --- |
| V1 | 一个父组在一个任务内只能有一个 `split.primary`；重复出现必须合并或明确不同角色 |
| V2 | `task_real` 先登记标签状态与证据；无标签素材可预分配 split 并记 `to_label`，补齐本次监督标签后再作为正式图签输出 |
| V3 | 背景带任务范围内的语义证据。配对 checker 接受 `empty_negative` 的 `confirmed_empty`、`candidate_empty` 和 `candidate_complete_negative` 声明。正式空负例监督由接收任务按其识别范围检查整帧并确认证据，目录分类仅用于发现候选。 |
| V4 | 派生数据继承父组关系，跨 split 使用沿用允许的同源范围。 |
| V5 | 固定图案、生成文字、Blender 母场景用 `shared`；同一场景跨 split 记为同源，不算独立父组 |
| V6 | 80% 父组份额用于提示留源空间和报告当前覆盖；父组少或单父组时按现有可行来源继续制作，并标明同源评估范围。 |
| V7 | `val_test_mode: reserve_only` 时不得在 `pairing` 写张数、预算或配额 |
| V8 | 使用 `same_source_allowed_in` 的 split 必须在报告中标为同源，不能声称独立场次 |
| V9 | 同一任务内逐字节完全重复的样本不得跨 train/val/test |
| V10 | 跨任务复制同一图片允许，但每个任务保留自己的标签、类别表和用途；父组按任务分别计，不重复计独立来源 |
| V11 | 计划列出每个 split 实际消费的父组；跨 split 同源使用明确登记，独立来源评估仅使用未接触父组。 |
| V12 | 单父组可先制作 train 与标明同源的 val/test。需要独立来源结论时登记新场次或未接触父组；计划补采用 `capture_id` 关联采集请求。 |

## 4. 任务级配对矩阵

### 4.1 汇总

| 任务 | 任务实拍父组（候选/已审） | 三向拆分（提议） | 背景父组（候选/已审） | 三向拆分（提议） | 补采 | 主要动作 |
| --- | --- | --- | --- | --- | --- | --- |
| person | 5–6 / 2–3 | 可行（2/2/2） | 10 / 2–3 | 可行（3/2/3） | 无 | 补标、审背景 |
| e_bike | 15 / 5 | 直接可行；掩膜仅 2 组 | 3 / 1 | 1/1/1 但仅 1 组合格 | 无 | 补标、提掩膜 |
| fire | 3 楼面 + 8 视频 / 3 照片 | 按楼面内批次拆 | 多 / 部分 | 可行 | 无 | 补审、补标 |
| building_name_paper | 1–2 / 1 | 不可 | 3+ / 1 | 可行 | 条件性 | 补标、补审 |
| sign | 1 / 1 | 不可 | 4 / 4 | 可行（2/1/1） | **2 场** | 补采 |
| waste_bin_state | 8 / 0 | 可行（3/2/3） | 1 / 1 | 不可 | 无 | 补标、扩池 |
| waste_sticker | 1 / 1 | 不可 | 6 / 6 | 可行（2/2/2） | **2 场** | 为独立实拍预留场次，扩充可用贴纸背景 |
| meter | 5 / 0 | 标签后可行 | 1–3 / 1 | 扩池后可行 | 条件性 | 补标、扩池 |

“已审”是当前有审查或选定证据的父组；不是实际产量。详细容量见[父源容量与分配可行性](PARENT_SOURCE_CAPACITY.md)。

### 4.2 person

| ID | 类别 | 来源/父组 | 可用量 | 角色/用途 | 建议 split | 状态/动作 |
| --- | --- | --- | ---: | --- | --- | --- |
| tr_person_standees | task_real | `person_standees.mp4` | 1,502 帧/12 段 | direct + composite target | train | 待标 |
| tr_full_scene_person | task_real | `full_scene` 人物段 | 14 审查帧 | composite target | train（与背景同视频） | 待标 |
| tr_scan_012 | task_real | `012` 场地扫描 | 49 审查帧 | target/背景 | val | 待标 |
| tr_car_real_person | task_real | CarReal 人物段 18 段 | 2,773 帧中 18 段 | direct + composite target | test | 待标 |
| tr_crowd_photos | task_real | `密集人群` 5 张 | 5 | target candidate | test | 待审 |
| tr_site_person | task_real | site_capture 车载图 | 68 条观测 | target candidate | val | 待审 |
| bg_full_scene | background | `full_scene.mp4` | 11,378 帧 | composite background | train | candidate_empty；yaml 已移除 empty_negative，转 to_review |
| bg_001 | background | `空场与场景物料_001.mp4` | 281 帧 | composite background | train | candidate_empty；yaml 已移除 empty_negative，转 to_review |
| bg_site_003/005 | background | 共享池 003、005 | 各 20 帧 | composite background | val | 待审 |
| bg_site_009 | background | 共享池 009 | 20 帧 | composite background | train | 待审 |
| bg_ebike_003 | background | e_bike 003 | 20 帧 | composite background | test | 待审 |
| bg_waste_004 | background | waste_bin 004 | 20 帧 | composite background | test | 待审（人物 5 帧已拒） |
| bg_car_real | background | CarReal 帧 | 2,773 帧 | composite background | test | 待审 |
| occl_008/009 | background | e_bike 008/009 | 2 父视频/4 掩膜 | occluder | 008/009 train | 待标。val/test 可从 005/006 选取，或登记同源预留 |
| ds_identity_18 | data_source | 16 社区 + 2 非社区图案 | 18 | target foreground | shared | 固定物料 |
| bl_arena_master/detailed | blender | 共享 arena | 2 场景 | background_scene | shared | 同源记录 |

上表记录 2026-11-15 草案对人物父组的用途安排。已落盘 split 和当前出包见[生产进度](DATASET_PRODUCTION.md#当前生产进度)；空负例监督按逐图任务范围的无目标证据选择。

### 4.3 e_bike

| ID | 类别 | 来源/父组 | 可用量 | 角色/用途 | 建议 split | 状态/动作 |
| --- | --- | --- | ---: | --- | --- | --- |
| tr_bike_008 | task_real | 008 倒伏平视 | 15 审核帧 | target/mask | train | 待标 |
| tr_bike_009 | task_real | 009 直立平视 | 12 审核帧 | target/mask | train | 待标 |
| tr_bike_003 | task_real | 003 场地多车低视角 | 6 审核帧 | target | val | 待标 |
| tr_scan_012 | task_real | 012 场地扫描 | 49 审核帧 | target | test | 待标 |
| tr_car_real_bike | task_real | CarReal 电动车段 16 段 | 帧源 | direct target | test | 待标 |
| tr_site_bike | task_real | site_capture 60 条观测 | 车载图 | direct target | val | 待审 |
| tr_bike_005_006 | task_real | 005 直立低视角 / 006 倒伏俯角 | 2 视频 | mask source | val | 待提掩膜；多视频合并项，需拆父组 |
| tr_bike_004_007_010_011 | task_real | 其余单车环绕 | 4 视频 | mask source | test | 待提掩膜；多视频合并项，需拆父组 |
| bg_full_scene | background | `full_scene` 4 帧 | 4 | composite background | train | ready |
| bg_car_capture | background | CarReal 6 张任务点照片 | 6 | composite background | val | 待审 |
| bg_legacy_001 | background | 001 的 2 帧 | 2 | composite background | test | 待审 |
| bg_site_003/005/009/waste_004 | background | 共享池 | 80 帧 | composite background | val | 待审；yaml 实际 split |
| ds_mask_from_008/009 | data_source | 21 掩膜 | 4 精修 | target foreground | 继承 008/009 | 派生 |
| bl_arena | blender | arena master/detailed | 2 场景 | background_scene | shared | 同源 |

### 4.4 fire

| ID | 类别 | 来源/父组 | 可用量 | 角色/用途 | 建议 split | 状态/动作 |
| --- | --- | --- | ---: | --- | --- | --- |
| tr_facade_A_photos_train/val/test | task_real | 开心大厦 25 张 | 25 | direct/composite target | train/val/test 按批次 | 待标 |
| tr_facade_B_photos_train/val/test | task_real | 锦绣大厦 18 张 | 18 | 同上 | train/val/test 按批次 | 待标 |
| tr_facade_C_photos_train/val/test | task_real | 晨光大厦 24 张 | 24 | 同上 | train/val/test 按批次 | 待标 |
| tr_facade_video_A_001/A_002/B_001_002/B_003/B_004/C_001/C_002 | task_real | 8 段楼宇视频 | 8 | target/geometry | 按视频批次 | 待标 |
| tr_fire_review_pool | task_real | 63 条审核候选（17 视频） | 63 | direct/composite target | 按视频分 train/val/test | 待审 |
| tr_car_real_fire | task_real | CarReal 火焰段 9 段 | 帧源 | target | test | 待标 |
| ds_pattern | data_source | `image1.png` | 1 | target foreground | shared | 固定图案 |
| bg_site_003 | background | 共享池 003 | 20 帧 | composite background | train | 待审 |
| bg_site_005/012 | background | 共享池 005、012 | 40 帧 | composite background | val | 待审 |
| bg_site_009/ebike_003/waste_004 | background | 共享池 | 60 帧 | composite background | test | 待审 |
| bl_three_buildings | blender | 三楼合景 | 1 场景 | target scene + background | shared | 同源 |

无贴楼面照片并入 task_real 的 `tr_facade_*_photos_*`，CarReal 火焰段为 `tr_car_real_fire`；不再单列 `bg_facade_photos`/`bg_car_real`。背景 split 以 source.yaml 的 `bg_site_003=train`、`005/012=val`、`009/ebike_003/waste_004=test` 为准。

注意：楼面 A/B/C 是组合轴，不允许把整栋楼整体分给单一 split；要在每个楼面内按照片/视频批次拆。当前只选 3 张照片（每楼 1 张），若没有第二批次证据，val/test 只能先用同源或补拍；不得用 A→train、B→val、C→test 的分配，否则 train 丢失 B/C 轴覆盖。

### 4.5 building_name_paper

| ID | 类别 | 来源/父组 | 可用量 | 角色/用途 | 建议 split | 状态/动作 |
| --- | --- | --- | ---: | --- | --- | --- |
| tr_full_scene_paper | task_real | `full_scene` 5 帧完整 | 5（2 正 + 3 负） | direct target | train | 完整候选 |
| tr_car_real_paper | task_real | CarReal 6 条/3 纸框 | 6 | target/geometry | val | 待标 |
| tr_review_57 | task_real | 57 个审核文件 | 57 | target candidate | 按视频分 train/val/test | 待审/待标 |
| ds_generated_text | data_source | 生成文字 | 生成器 | OCR target | shared | 与纸框配对 |
| bg_buildings | background | A/B/C 楼 | 3 楼 | composite background | 按批次 | 待审 |
| bg_full_scene | background | `full_scene` | 帧源 | composite background | train | 待审 |
| bg_site_pool | background | 共享池 | 120 帧 | composite background | val/test | 待审 |
| bl_three_buildings | blender | 三楼合景 | 1 场景 | target scene | shared | 900 计划已写 |

注意：楼面是组合轴；当前 source.yaml 把三栋楼整体分到 train/val/test 会与任务轴冲突（train 只剩 A 楼），编译前需按楼内照片/视频批次重新拆。本表不据此修改 split。

### 4.6 sign

| ID | 类别 | 来源/父组 | 可用量 | 角色/用途 | 建议 split | 状态/动作 |
| --- | --- | --- | ---: | --- | --- | --- |
| tr_sign_video | task_real | 牌具转面视频 | 8 掩膜视图 | target/mask | train（6 掩膜） | 已选 |
| tr_sign_capture_val | task_real | 新场次 A | 待采 | target/mask | val | 待采 |
| tr_sign_capture_test | task_real | 新场次 B | 待采 | target/mask | test | 待采 |
| ds_mask | data_source | 8 掩膜 | 8 | target foreground | 继承视频 | 派生 |
| bg_site_003/005 | background | 共享池 | 40 帧 | composite background | train | 待审 |
| bg_photo_batch | background | site_capture 车载图 | 2 张选定/160 张候选 | composite background | val | 待审 |
| bg_ebike_003 | background | e_bike 003 | 20 帧 | composite background | test | 待审 |
| bg_site_009/012 | background | 共享池 009、012 | 复核按逐图回执 | composite background | 本批 Val | 已完成分集，已有独立 Val 小样 |
| bg_waste_004 | background | bins004 同父视频 | 本次 21 张合格帧 | composite background | Val（已分配） | 跨任务已有 Val 接触 |
| bl_arena_detailed | blender | arena detailed | 1 场景 | background_scene | shared | 无牌具三维模型 |

禁行牌旧 inventory 的155条帧已复核，147条合格底图分为Train70、Val77，原有585张Train底图继续可用；`bins004`整父来源归Val。当前全量Train2500、Val300，分别2000／275框、500／75空负，使用344／67张底图或原帧、15／4个背景父组。集内及跨集无相同RGB图，背景父组无跨集交叉；前景仍共享同一牌具视频，Val为受控诊断。440张已接受小样保留。另4条主体帧合成用途受阻、4条落点未知的用途按现行资格使用。[全量审计](../datasets/annotated/sign/full_dataset_audit.json) · [155帧复核回执](../datasets/annotated/common/review/sign_background_155_20261118/recheck_receipt.json) Test另使用20张既有停车视频帧，300张／225框／75空负；19条历史待验证资格已补齐当前合成接触检查。其背景父组与本批Train／Val不交叉，已有跨任务及旧检查接触，前景共享，评价为回归诊断。

### 4.7 waste_bin_state

| ID | 类别 | 来源/父组 | 可用量 | 角色/用途 | 建议 split | 状态/动作 |
| --- | --- | --- | ---: | --- | --- | --- |
| tr_bin_001 | task_real | 四色桶 001 | 1 视频 | target | train | 待标 |
| tr_bin_003 | task_real | 四色桶 003 | 1 视频 | target | train | 待标 |
| tr_bin_005 | task_real | 灰桶关盖 005 | 1 视频 | target | train | 待标 |
| tr_bin_004 | task_real | 四色桶 004 | 1 视频 | target | val | 待标 |
| tr_bin_006 | task_real | 灰桶开盖 006 | 1 视频 | target | val | 待标 |
| tr_bin_007 | task_real | 浅色地面 007 | 1 视频 | target | test | 待标 |
| tr_user_photos | task_real | 用户多视角照片 14 | 14 | target/reference | test | 待审 |
| tr_motion_video | task_real | 运动方式 | 1 视频 | target | test | 待标 |
| ds_bin_blender | data_source | `waste_bin_pbr.blend` | 1 场景 | target/background | shared | 19 状态计划 |
| bg_full_scene | background | `full_scene` 14 帧 | 14 | composite background | train（同源允许 val/test） | backgrounds.json 10 train/2 val/2 test；pairing 只登记 train 主 split，val/test 非独立父组 |
| bg_site_003/005/009/012/ebike_003 | background | 共享池 | 100 帧 | composite background | val/test | 待审 |

### 4.8 waste_sticker

| ID | 类别 | 来源/父组 | 可用量 | 角色/用途 | 建议 split | 状态/动作 |
| --- | --- | --- | ---: | --- | --- | --- |
| tr_p22_capture | task_real | P22 + V 视频，同一采集组 | 32 张完整、11 张部分 | 同源回归 | 单列 | 另选四张回归，002016 用于历史候选包 |
| tr_sticker_capture_val | task_real | 新场次 A | 待采 | direct/composite target | val | 待采 |
| tr_sticker_capture_test | task_real | 新场次 B | 待采 | direct/composite target | test | 待采 |
| bg_001 | background | 四色桶 001 | 1 张本批实拍桶面 | surface/background | train | 已用于 45 张桶面叠贴 |
| bg_004 | background | 四色桶 004 | 20 张合格桶面帧 | surface/background | Val（已整组绑定） | 已用于 80 张本域桶面合成 |
| bg_003 | background | 四色桶 003 | 1 视频 | surface/background | train | 已选 0005 桶面 |
| bg_005 | background | 灰桶关盖 005 | 1 视频 | background | val | 待标 |
| bg_007 | background | 浅色地面 007 | 1 视频 | surface/background | test | 待标 |
| bg_006 | background | 灰桶开盖 006 | 1 视频 | background | test | 待标 |
| ds_surface_0002 | data_source | 001 的 0002 表面 | 1 表面（红桶） | surface | 继承 bg_001 → train | 派生 |
| ds_surface_0005 | data_source | 003 的 0005 表面 | 1 表面（红桶） | surface | 继承 bg_003 → train | 已用于当前完整 Train 的 45 张桶面叠贴 |
| ds_surface_007 | data_source | 007 的非红桶面 | 待提 | surface | 继承 bg_007 → test | 待提取 |
| ds_24_patterns | data_source | `SourcePic` 24 张 | 24 | target foreground | shared | 固定 24 类 |
| bl_sticker_animated | blender | 统一工程的贴纸场景和场地 | 同一 Blender 基础资产 | target scene + background | 基础资产共享，布局按 split 留出 | 旧 26 状态单列诊断 |

桶面继承父组的 split。旧 Train 子批使用 0002（父组 001）与 0005（父组 003）。2026-11-17 核对时，整桶混合已有两张合格 Train 实拍场地帧，来自 `full_scene.mp4` 与 `空场与场景物料_001.mp4`。Test 已整组绑定 `junction_005` 的 13 张合格实拍场地帧，用于整桶混合。Test 场地排除已用于 Train 的 `full_scene.mp4`。拍摄原帧、布局和产出图以各自身份计数。[分配回执](../datasets/annotated/waste_bin/whole_bin_ground_reuse_20261117/production_split_assignment_receipt.json) · [垃圾贴纸来源](../datasets/scripts/waste_sticker/config/source.yaml)

### 4.9 meter

| ID | 类别 | 来源/父组 | 可用量 | 角色/用途 | 建议 split | 状态/动作 |
| --- | --- | --- | ---: | --- | --- | --- |
| tr_car_real_meter | task_real | CarReal 45 帧 | 45 | direct target | train | 待标（无完整 Pose） |
| tr_site_meter | task_real | site_capture 7 帧 | 7 | direct target | val | 待标 |
| tr_archive_meter | task_real | 归档 2 图 + 任务点 1 | 3 | direct target | test | 待标 |
| tr_meter_capture_val/test | task_real | 新场次（条件性） | 待采 | direct target | val/test | 条件性补采 |
| ds_station_blend | data_source | `station.blend` 12 针位 | 1 场景 | target scene | shared | 12 渲染状态 |
| bg_full_scene | background | `full_scene` 3 帧 | 3 | composite background | train | 已选 |
| bg_blocked_11 | background | 11 条 composite_background | 11 | composite background | unassigned（解除后分 val/test） | 待解除 blocked |
| bg_site_pool | background | 共享池 | 60 帧 | composite background | val/test | 待审 |
| bl_arena | blender | arena master | 1 场景 | background_scene | shared | 同源 |

### 4.10 来源组成

| 任务 | 目标数据源 | Blender 角色 | 直接实拍 / 实拍背景合成 / Blender | Blender 上限 |
| --- | --- | --- | --- | ---: |
| fire | 楼宇合景 + 固定火焰图案 | `primary` | 20/40/40 | 40% |
| building_name_paper | 三楼合景 + 生成文字 | `primary` | 20/40/40 | 40% |
| waste_bin_state | 桶模型 `waste_bin_pbr` | `primary` | 20/40/40 | 40% |
| waste_sticker | 贴纸动画场景、贴纸图案和合格桶面 | `primary` | 0/30/70 | 70% |
| meter | 参数化站房 | `primary` | 10/40/50 | 50% |
| person | `team_assets` 数字图片 | `background_only` | 30/40/30 | 40% |
| e_bike | 平视/环绕旋转掩膜 | `background_only` | 30/40/30 | 40% |
| sign | 牌具视频帧裁片与已有完整实拍 | `background_only` | 0.36/67.64/32 | 40% |

份额读取 [全量 train 分配配置](../config/full_train_allocation.yaml)。按当前标签和可行状态编排，实际占比随批次清单记录。场景的目标与背景组件分别登记在各任务 `source.yaml`。

## 5. 共享背景池扩用矩阵

代码：`T` = 该任务实拍，`B` = 背景，`-` = 不适用/未知；`tr/va/te` = 建议 split；`?` = 待审。

| 共享池父视频 | person | e_bike | fire | bnp | sign | waste_bin | waste_sticker | meter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 三栋楼_垃圾桶与路口_003 | B-tr | B-va | B-tr | B-tr | B-tr | B-va | B-? | B-va |
| 三栋楼_路口及垃圾桶_005 | B-va | B-va | B-va | B-va | B-tr | B-va | B-te | B-va |
| 三栋楼侧向扫描_009 | B-tr | B-va | B-va | B-va | B-va | B-te | B-? | B-te |
| 人物立牌正背面_电动车_012 | T-va | T-te | B-te | B-te | B-te | B-te | B-? | B-te |
| 场地多辆电动车_较低视角_003 | B-te | T-va | B-tr | B-tr | B-te | B-te | B-? | B-te |
| 四色桶较低视角_004 | B-te | B-va | - | - | B-va | T-va | B-tr | - |

矩阵记录初始用途提议。实际使用从各任务 `source.yaml` 和当前逐图记录查询。复用已有符合范围的记录；仅对本批选中且缺证据的素材，在入库时补相应任务字段。父组及 split 归属按实际记录。[背景图入库与复用](DATASET_DESIGN.md#背景图入库与复用)

## 6. 独立场次补采清单

本节记录未来可能使用的独立采集占位。sign 保留补采计划。垃圾贴纸按[现行四路来源和分集设计](DATASET_DESIGN.md#垃圾贴纸数据设计)推进，识别失败需要真实成像条件时再定向补拍。meter 与 building_name_paper 先使用现有素材，按识别失败及实际需要决定补拍。

| ID | 任务 | 目标 split | 拍摄对象 | 场景/摆位 | 视角 | 光照 | 数量/时长 | 真值 | 保存位置 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cap_sign_val_20261116 | sign | val | 禁止直行牌具 | 与现有视频不同的路口/白墙摆位 | 正、斜、侧、背 | 正常、暗、亮 | 2–3 分钟视频或 100–200 张照片 | 牌具框、视面、截断、摆位 | `Archive/Assets/captures/sign/capture_20261116_val/` |
| cap_sign_test_20261116 | sign | test | 同上 | 第三种摆位/背景 | 正、斜、侧、背 | 正常、暗、亮 | 同上 | 同上 | `Archive/Assets/captures/sign/capture_20261116_test/` |
| cap_sticker_val_20261116 | waste_sticker | val | 固定 24 类物品贴纸，可独立拍摄或复刻每桶正面下方一张贴纸的现场画面 | 与 P22 不同的布置、背景和贴纸位置 | 正、斜 | 正常、暗 | 30–60 帧/2–3 段短视频 | 每贴类别与可见框，有桶时另记可核桶 ID、桶色 | `Archive/Assets/captures/waste_sticker/sticker_capture_20261116_val/` |
| cap_sticker_test_20261116 | waste_sticker | test | 同上 | 第三种布置、背景和贴纸位置 | 正、斜 | 正常、暗、亮 | 同上 | 同上 | `Archive/Assets/captures/waste_sticker/sticker_capture_20261116_test/` |
| cap_meter_val_20261116 | meter | val | 实物温湿表/站房 | 与现有视频不同安装/读数 | 正、斜 | 正常、反光 | 2 段 1–2 分钟视频或 50–100 张 | 8 点、读数、可见性 | `Archive/Assets/captures/meter/capture_20261116_val/` |
| cap_meter_test_20261116 | meter | test | 同上 | 第三种读数/反光 | 正、斜 | 正常、强反光 | 同上 | 同上 | `Archive/Assets/captures/meter/capture_20261116_test/` |
| cap_bnp_val_20261116 | building_name_paper | val | 楼顶纸条（自定义名称） | 与现有纸框不同的楼/位置 | 正、斜 | 正常、暗 | 30–60 张/2 段视频 | 纸框、全文、楼号 | `Archive/Assets/captures/building_name_paper/capture_20261116_val/` |
| cap_bnp_test_20261116 | building_name_paper | test | 同上 | 第三种楼/名称 | 正、斜 | 正常、暗 | 同上 | 同上 | `Archive/Assets/captures/building_name_paper/capture_20261116_test/` |

说明：

- 补采占位父组在 `source.yaml` 中用 `capture_session: planned` 和 `capture_id` 关联本表；采集完成并登记证据后才改 `proven`。派生掩膜等未生成的数据源同样标 `action: to_capture`，不得提前标 `confirmed`。
- 用已合格的共享底图、Blender 和实拍素材推进合成和训练，入库端同期补判受影响的分支。实际漏检、混淆或完整任务失败需要真实条件时再定向补拍。近邻帧与同母场景渲染按实际父来源和用途记账。
- `meter`、`building_name_paper` 标为条件性：先在现有视频/照片中补标 Pose、纸框与文字；如果审查后仍凑不出独立 val/test 父组，再按上表补采。
- 每场采集保留原图/原视频、拍摄参数、摆位草图、真值文件和父组 ID；原件按父批次放入 `Archive/Assets/captures/`，任务标注写到 `datasets/annotated/`，选用的目标或背景工作图进入对应的 `raw/` 目录。
- 补采不改变现有 train 配额；val/test 张数仍待后续确定。

## 7. 跨任务复用与相同图片规则

1. **同一任务内**：父组有主 split，帧及派生保留父组关系；明显注明的同源复用以实际 split 和配对声明为准。逐字节完全重复的样本不得跨 train/val/test。同视频相邻帧及视觉近似帧的关系用于抽样和独立性统计，有效任务标签与用途资格仍按帧核对。
2. **跨任务**：允许复制同一图片到多个任务；每个任务保留自己的标签、类别表、用途和审核状态。父组按任务分别计，不因复制变成新的独立来源。
3. **同图不同模型**：同一画面可以用人物标签训练 person，也可以给 e_bike 提供背景或遮挡物。每个模型读取自己的标签文件。
4. **父源别名与历史接触**：用父视频散列、迁移记录和实际批次回执核齐同一物理父源的路径。当前 split 和技术资格随明确的本批用途登记，train、检查或旧评估中已使用的记录随父源保留。既有接触的 test 用途用于诊断，独立评估按未接触的来源计。
5. **同源跨 split**按当前用途登记 `same_source_allowed_in`，在批次清单与报告里写明共享的是视频、场次、物料、背景还是其他父素材。
6. **评估解释**核对待评模型的训练和既往诊断接触，按实际留出的维度解释。跨任务比较记录已知共享的父来源和角色。
7. **80% 账本**：按任务×角色统计 train 父组占比。跨任务汇总时按 Blender 母场景归并实际用量。

## 8. 贝叶斯与计划接入

- 可执行计划声明 `target_split: train | val | test`，`pairing.plan_bindings` 关联该 split 的计划。
- `source_mix` 保存 train 的来源配额参考。编制各 split 时按实际来源和标签容量分配。
- 计划 `strata` 的每一条至少声明：父组 ID、数据源、背景、组合轴状态、标签、配额上下限；分配器只在已声明 strata 内分额度。
- 新编计划时运行配对校核，重点处理本次引用的父组、标签范围及已知冲突。预算按当批可行组合编制，80% 父组参考用于提示留源和报告覆盖。
- 预留记录描述来源用途，预算写在相应执行计划里。人物已实际导出 train/val/test，其他任务按自己的计划和产物记录。
- `state_coverage` 统计在任务目标及组合轴配置中明确登记的计划。实际覆盖与产量查具体批次的导出清单。
- 跨任务 Blender 账本待实现；同一母场景（arena master、three_buildings、station、waste_bin_pbr）跨任务合计用量要按父场景汇总。

### 8.1 计划对齐与 checker 边界

`source.yaml` 的 `pairing.plan_alignment` 记录现有计划与 pairing split 的冲突；`status: stale` 标出历史计划的配对冲突。重编受影响的状态或修订该状态的 split，其他已合格状态继续生产。当前记录：

- person 已登记 `full_train_plan_3000.json`、`val_plan_300.json` 和 `test_plan_300.json`。现行批次按已记录的背景父组切分，其他角色保留共享来源。
- sign：当前全量背景按15个Train与4个Val及1个Test父组分集，具体路径见full_train_plan.json、full_val_plan.json、full_test_plan.json与全量清单；牌具前景共享，受控诊断范围随训练来源索引记录。
- waste_sticker：四路完整数据已导出 1,800／240／240 张。Train 桶面使用 `bg_001`、`bg_003`，Train 整桶场地使用 `空场与场景物料_001`，Val 桶面使用 `bg_004`，Test 整桶场地使用 `junction_005`。共享固定图案和 Blender 基础资产按同源诊断报告。[导出审计](../datasets/exports/waste_sticker/audit.json)
- waste_bin_state：`backgrounds.json` 的 10 train/2 val/2 test 全部同属 `full_scene.mp4`；pairing 只登记 train 主 split 和 `same_source_allowed_in`，val/test 不是独立父组。
- e_bike：组合来源计划给 `bg_car_capture`（val）、`bg_legacy_001`（test）排了 train 份额，尚未重编译为逐 split 计划。

`source_pairing.py --check` 读取八任务的配对声明，检查结构、角色枚举、负例角色状态、capture、路径、来源组成及 `plan_alignment` 字段。格式有效的 `stale` 可通过。V11 的实际计划引用、逐图标签和跨 split 字节重复按受影响的实际批次核对。[现行入口限制](DATASET_PRODUCTION.md#现行入口限制与处理)

其他未决：

- 一个 `parent_video` 项含多段视频时，按实际父视频统计覆盖与 split。电动车 `tr_bike_005_006`、`tr_bike_004_007_010_011` 和火焰 `tr_facade_video_B_001_002` 的合并条目，按本批实际引用的视频逐一记账。
- 楼名覆盖按实际楼面、布置和父源核算。按已合格场景运行可行分支，需独立楼面评估时按相应批次留出并记录覆盖范围。
- 配对声明使用 `source_pairing.py --check` 校核。父组诊断查询使用 [source_ledger](BACKGROUND_SOURCE_LEDGER.md)。

## 9. 实施顺序

1. 从现有 `source.yaml`、任务标签和查询索引选本批可用来源，记录角色、split、已知同源和判定依据。
2. 已录入的适用背景直接参与配额。所需任务证据有缺项时，使用现有判定器或标签入口更新该项。确有本批所需的来源缺口时使用对应补采建议。
3. 编译受影响的 split 计划，运行配对声明检查并核对本批实际引用。可行分支按任务入口生成或打包，清单记录组归属和产量。

## 10. 未决项

- 第 4 节的逐任务 split 提议是否接受；特别是 person 用 012 作 val、CarReal 作 test，fire 按楼面内批次拆，waste_bin 视频 3/2/3。
- 电动车组合计划仍引用 val、test 的父组。桶开闭三个 split 共用 `full_scene.mp4`，按同源计划解释。编制相关批次时核对各自的 `plan_alignment`。
- bnp 三栋楼整体分 split 与楼面组合轴冲突，需按楼内批次重新拆；e_bike/fire 的多视频合并 parent_video 项需拆父组。
- meter、building_name_paper 的补采是否必要；若现有素材补标后能凑出独立父组，可取消。
- 共享池矩阵中 `?` 的背景资格需要逐任务审查后才能确定。
- 跨任务是否需要"全局未见场次"预留；若需要，当前按任务独立分 split 的方案不足。
- 各任务的 val/test 张数按执行计划登记。尚无预算的任务先记录父组预留。
