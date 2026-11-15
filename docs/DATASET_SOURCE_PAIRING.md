# 数据集父子源配对与分集预留设计

> 状态：结构设计草案，2026-11-15。用户已决定：配对写入各任务 `source.yaml`；单父组任务补采独立场次；扩用共享背景池；val/test 先只预留父组、不定张数。本文提取需求、定义变量空间与结构、给出逐任务矩阵和补采清单；`pairing` 草案已按本结构写入八任务 `source.yaml`（`status: draft`），**未修改计划或配额**。
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

项目既有约束继续适用：不按帧数冒充独立来源；派生数据继承父来源；同一任务内不得把逐字节完全重复的样本分到不同监督集合；跨任务复用需记录实际范围；不提交数据、权重或凭据；生产和训练继续保持暂停。

## 2. 变量空间

### 2.1 三层概念必须分开

| 层 | 回答的问题 | 本设计中的角色 |
| --- | --- | --- |
| 父源类别 | 这是本任务实拍，还是背景？ | 决定 split 预留、80% 上限、val/test 独立性 |
| 组合轴状态 | 在同一 split 内要覆盖哪些视角、光照、颜色、楼面等？ | 决定合成状态和覆盖矩阵，不决定 split |
| 数据源角色 | 合成时它是目标、遮挡物还是背景？ | 决定标签和图层，不单独算父组 |

一个容易混淆的点：**遮罩、裁片、掩膜是数据源，不是新的父组**。它们继承原父组的 split；从 train 父组裁出的掩膜不能拿到 test 合成，除非明确记为同源复用。

另一条判定规则：**父源类别按画面内容和任务用途判定，不按所在目录判定**。同一父视频对任务 A 可以是任务实拍（画面含 A 的目标），对任务 B 是背景（画面不含 B 的目标）；本任务目录里的空场属于背景，其他任务 raw 里含本任务目标的帧也可以属于任务实拍。目录只用于查找，不决定类别。

同一父视频可以同时提供任务实拍帧和背景帧（例如 `full_scene` 的人物段与空场段），但整段父视频只归一个 split；不同角色不改变父组归属，也不能把同一视频的不同帧分到不同 split 再声称独立。

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
| 计划绑定 | train/val/test 对应计划路径；当前 val/test 为空 |
| 采集动作 | `selected`、`reserved`、`to_review`、`to_label`、`to_extract`、`to_capture`、`blocked` |

### 2.3 来源 × 用途矩阵

| | 直接使用 | 合成使用 |
| --- | --- | --- |
| **任务实拍** | 直接实拍图分到各 split | 任务实拍前景/掩膜 + 背景合成 |
| **背景** | 确认空场/负例图（如 300 张空街） | 数据源（图案/数字图/Blender 目标）+ 背景合成 |
| **Blender 母场景** | 纯渲染背景或负例 | 背景层，或目标层（楼宇、站房、桶、贴纸场景） |

30% 直接实拍 / 40% 实拍背景合成 / 30% Blender 是**基线**；其中 fire、building_name_paper、waste_bin_state、waste_sticker、meter 已按下一段调整，其余任务保持基线。split 分配是另一条轴：**先按父组把来源分到 train/val/test，再在每个 split 内跑组合轴和贝叶斯**。

2026-11-15 用户确认采用：fire、building_name_paper、waste_bin_state 采用 **20% 直接实拍 / 40% 实拍背景合成 / 40% Blender**，waste_sticker、meter 采用 **10% / 40% / 50%**；person、e_bike、sign 保持 30/40/30，目标数据源分别用数字图片、旋转掩膜、视频帧裁片。详见 4.10。

## 3. `source.yaml` 配对结构

### 3.1 放置与权威

- 每个任务 `config/source.yaml` 新增顶层 `pairing`；它是**父子源与 split 预留的权威记录**，计划编译时消费它。
- 现有 `current_selection`、`production_plan` 保留；`pairing` 不替代批次清单和计划，只记录"哪个父组归哪个 split、扮演什么角色、证据在哪"。批次清单里的 `task_use_status` 继续保存逐图用途；`pairing` 汇总到父组级，二者用 `manifest` / `record_index` 关联，不另建第二来源台账。
- 生成器只读取自己认识的键；新增 `pairing` 不改变现有入口。多数入口把 `source.yaml` 声明为配置输入，新增字段会改变输入哈希，使相应旧生产回执不再复用；未声明 `source.yaml` 的入口不受影响。生产暂停期间可接受，编译新计划时按实际入口重新登记。
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
        - datasets/raw/site_capture_20261112/videos/person_standees.mp4
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
        - datasets/raw/site_capture_20261112/videos/full_scene.mp4
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
      path: datasets/raw/person/team_assets/社区人员/1.png
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
      save_to: datasets/raw/sign/capture_20261116_val/
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
| `split.same_source_allowed_in` | 可选 | 只在单父组无法三向时使用；列出的 split 必须按同源解释 |
| `parent_group` | 派生数据源必填 | 指向 `task_real` 或 `background` 的父组 ID；固定图案/生成文字为 `null` |
| `plan_bindings` | 可选 | 该 split 后续编译到哪份计划；当前 val/test 为空 |
| `plan_alignment` | 可选 | 现有计划与 pairing split 的对齐状态：`status: aligned/stale/unknown`、`note` 和 `conflicts`；`stale` 表示现有计划仍消费其他 split 的父组，编译前必须重编或改 split |
| `plan_alignment.conflicts` | 可选 | 逐条记录 `parent`、`plan`、`pairing_split`、`plan_use`；只作待决标记，不修改计划 |
| `captures` | 可选 | 独立场次补采请求；字段按 6 节 |
| `captures.status` | 是 | `proposed` / `conditional` / `confirmed`；与 `task_real` 占位的 `capture_id` 关联 |

### 3.4 校核规则

| ID | 规则 |
| --- | --- |
| V1 | 一个父组在一个任务内只能有一个 `split.primary`；重复出现必须合并或明确不同角色 |
| V2 | `task_real` 必须有标签状态和证据；无标签的实拍只能记为 `to_label`，不能直接进计划 |
| V3 | 背景必须有审查状态；`confirmed_empty` 才能作空负例（`empty_negative`），`candidate_empty` 只能作待审背景或显式 `action: to_review`，不得提前赋予空负例角色；`contains_target` 只能作已标目标背景 |
| V4 | 派生数据源必须继承父组 split；不能单独改到另一个 split |
| V5 | 固定图案、生成文字、Blender 母场景用 `shared`；同一场景跨 split 记为同源，不算独立父组 |
| V6 | 每个任务×角色的 train 父组数不超过合格父组的 80%；不足 5 个时报告实际数量 |
| V7 | `val_test_mode: reserve_only` 时不得在 `pairing` 写张数、预算或配额 |
| V8 | 使用 `same_source_allowed_in` 的 split 必须在报告中标为同源，不能声称独立场次 |
| V9 | 同一任务内逐字节完全重复的样本不得跨 train/val/test |
| V10 | 跨任务复制同一图片允许，但每个任务保留自己的标签、类别表和用途；父组按任务分别计，不重复计独立来源 |
| V11 | 每个 split 的合成计划只能消费分配给该 split 的父组；当前尚无自动 checker，编译前专项核对 |
| V12 | 单父组任务若要求独立 val/test，必须登记 `captures`；未补采前只能同源预留；占位父组用 `capture_id` 关联采集请求 |

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
| waste_sticker | 1 / 1 | 不可 | 6 / 6 | 可行（2/2/2） | **2 场** | 补采、扩表面 |
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
| occl_008/009 | background | e_bike 008/009 | 2 父视频/4 掩膜 | occluder | 008/009 train | 待标；用户确认 009 改 train，val/test 待 005/006 或同源预留 |
| ds_identity_18 | data_source | 16 社区 + 2 非社区图案 | 18 | target foreground | shared | 固定物料 |
| bl_arena_master/detailed | blender | 共享 arena | 2 场景 | background_scene | shared | 同源记录 |

person 的 `tr_*` 未完成标签已在 source.yaml 标 `action: to_label`；`bg_full_scene`/`bg_001` 的 `candidate_empty` 不再带 `empty_negative`，改 `to_review`。用户确认 `occl_009` 改为 train 后，`plan_alignment` 已对齐；val/test 仍待 005/006 掩膜或同源预留。空场 300 仍需 `confirmed_empty` 后才能按配额生产。

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
| bg_site_009/012/004 | background | 共享池 | 60 帧 | composite background | val/test | 待审 |
| bl_arena_detailed | blender | arena detailed | 1 场景 | background_scene | shared | 无牌具三维模型 |

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
| tr_p22_capture | task_real | P22 + V 视频 | 42 帧/1 可用 | direct target | train | 已选 1 帧 |
| tr_sticker_capture_val | task_real | 新场次 A | 待采 | direct/composite target | val | 待采 |
| tr_sticker_capture_test | task_real | 新场次 B | 待采 | direct/composite target | test | 待采 |
| bg_001 | background | 四色桶 001 | 1 视频 | surface/background | train | 待标 |
| bg_004 | background | 四色桶 004 | 1 视频 | surface/background | train | 待标 |
| bg_003 | background | 四色桶 003 | 1 视频 | surface/background | val | 待标 |
| bg_005 | background | 灰桶关盖 005 | 1 视频 | background | val | 待标 |
| bg_007 | background | 浅色地面 007 | 1 视频 | surface/background | test | 待标 |
| bg_006 | background | 灰桶开盖 006 | 1 视频 | background | test | 待标 |
| ds_surface_0002 | data_source | 001 的 0002 表面 | 1 表面（红桶） | surface | 继承 bg_001 → train | 派生 |
| ds_surface_0005 | data_source | 003 的 0005 表面 | 1 表面（红桶） | surface | 继承 bg_003 → val | 派生 |
| ds_surface_007 | data_source | 007 的非红桶面 | 待提 | surface | 继承 bg_007 → test | 待提取 |
| ds_24_patterns | data_source | `SourcePic` 24 张 | 24 | target foreground | shared | 固定 24 类 |
| bl_sticker_animated | blender | 动画场景 + arena | 2 场景 | target scene + background | shared | 26 test 状态 |

注意：表面是背景父组的派生数据源，必须继承父组 split。0005 表面来自 003 视频，所以 003 放 val 而不是 train；0002 表面随 001 放 train；非红桶面从 007 提，随 007 放 test。这样 train 背景父组为 001/004（2/6，33%），val 为 003/005，test 为 007/006，四色桶覆盖在每个 split 内保留。

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

### 4.10 来源组成调整（用户 2026-11-15）

| 任务 | 目标数据源 | Blender 角色 | 原来源组成 | 新来源组成 | Blender 上限 |
| --- | --- | --- | --- | --- | ---: |
| fire | 楼宇合景 + 固定火焰图案 | `primary` | 30/40/30 | 20/40/40 | 40% |
| building_name_paper | 三楼合景 + 生成文字 | `primary` | 30/40/30 | 20/40/40 | 40% |
| waste_bin_state | 桶模型 `waste_bin_pbr` | `primary` | 30/40/30 | 20/40/40 | 40% |
| waste_sticker | 贴纸动画场景 | `primary` | 30/40/30 | 10/40/50 | 50% |
| meter | 参数化站房 | `primary` | 30/40/30 | 10/40/50 | 50% |
| person | `team_assets` 数字图片 | `background_only` | 30/40/30 | 不变 | 40% |
| e_bike | 平视/环绕旋转掩膜 | `background_only` | 30/40/30 | 不变 | 40% |
| sign | 牌具视频帧裁片 | `background_only` | 30/40/30 | 不变 | 40% |

理由与边界：

- 五个任务的 raw 实拍中，完整标签或可用父组有限（fire 直接样本全 blocked、bnp 仅 5 帧完整、waste_bin 无逐桶真值、sticker 仅 1 个带贴采集组、meter 无完整 Pose）；对应 Blender 模型和场景已就绪，因此把 Blender 提到任务上限。
- 实拍背景合成仍保留 40%，用于跨背景多样性和负例；直接实拍降到 20% 或 10%，实际产量受可用标签限制，不能按份额虚构样本。
- person、e_bike、sign 的目标数据源分别保持数字图片、旋转掩膜、视频帧裁片；Blender 只作背景，不作为目标来源。这三个任务不提高 Blender 份额。
- `source.yaml` 的 `composition_policy` 记录上述方向；五个任务的 Blender 场景标 `priority: primary`，person/e_bike/sign 标 `priority: background_only`。目标与背景组件分别写入 `target_components` 和 `background_components`。
- `config/full_train_allocation.yaml` 的 `source_mix` 已按 20/40/40 与 10/40/50 调整，`docs/FULL_TRAIN_PLAN_DESIGN.md` 已重新生成；组配额未变，生产计划尚未编译。份额已由用户确认采用；后续调整另行记录，不改变组配额。

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

矩阵是候选用途，不代表已审可用，也不代表当前 `source.yaml` 的 split；每个 `B-*` 需要对应任务的空场/目标完整性审查，`T-*` 需要任务标签。实际 split 以各任务 `source.yaml` 和 §4 表为准；矩阵待逐任务审查后再回写，当前不一致不回写为既成事实。共享池扩用后，各任务背景父组数增加，val/test 才有独立父组可用。

## 6. 独立场次补采清单

按用户决定，单父组任务补采独立场次。以下为补采请求：sign、waste_sticker 状态 `proposed`，meter、building_name_paper 状态 `conditional`（先补标，仍不足再采），均未执行。

| ID | 任务 | 目标 split | 拍摄对象 | 场景/摆位 | 视角 | 光照 | 数量/时长 | 真值 | 保存位置 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cap_sign_val_20261116 | sign | val | 禁止直行牌具 | 与现有视频不同的路口/白墙摆位 | 正、斜、侧、背 | 正常、暗、亮 | 2–3 分钟视频或 100–200 张照片 | 牌具框、视面、截断、摆位 | `datasets/raw/sign/capture_20261116_val/` |
| cap_sign_test_20261116 | sign | test | 同上 | 第三种摆位/背景 | 正、斜、侧、背 | 正常、暗、亮 | 同上 | 同上 | `datasets/raw/sign/capture_20261116_test/` |
| cap_sticker_val_20261116 | waste_sticker | val | 四色桶 + 24 类贴纸（每桶正面一张） | 与 P22 不同的桶位/背景 | 正、斜 | 正常、暗 | 30–60 帧/2–3 段短视频 | 每贴类别、桶 ID、桶色、开闭、可见框 | `datasets/raw/waste_bin/sticker_capture_20261116_val/` |
| cap_sticker_test_20261116 | waste_sticker | test | 同上 | 第三种桶位/背景 | 正、斜 | 正常、暗、亮 | 同上 | 同上 | `datasets/raw/waste_bin/sticker_capture_20261116_test/` |
| cap_meter_val_20261116 | meter | val | 实物温湿表/站房 | 与现有视频不同安装/读数 | 正、斜 | 正常、反光 | 2 段 1–2 分钟视频或 50–100 张 | 8 点、读数、可见性 | `datasets/raw/meter/capture_20261116_val/` |
| cap_meter_test_20261116 | meter | test | 同上 | 第三种读数/反光 | 正、斜 | 正常、强反光 | 同上 | 同上 | `datasets/raw/meter/capture_20261116_test/` |
| cap_bnp_val_20261116 | building_name_paper | val | 楼顶纸条（自定义名称） | 与现有纸框不同的楼/位置 | 正、斜 | 正常、暗 | 30–60 张/2 段视频 | 纸框、全文、楼号 | `datasets/raw/fire/bnp_capture_20261116_val/` |
| cap_bnp_test_20261116 | building_name_paper | test | 同上 | 第三种楼/名称 | 正、斜 | 正常、暗 | 同上 | 同上 | `datasets/raw/fire/bnp_capture_20261116_test/` |

说明：

- 补采占位父组在 `source.yaml` 中用 `capture_session: planned` 和 `capture_id` 关联本表；采集完成并登记证据后才改 `proven`。派生掩膜等未生成的数据源同样标 `action: to_capture`，不得提前标 `confirmed`。
- `sign`、`waste_sticker` 是当前明确的补采项：不补采则 val/test 只能与 train 同源。每任务至少两场独立采集，分别用于 val 和 test；同一场内的不同帧/视图仍算同一父组。
- `meter`、`building_name_paper` 标为条件性：先在现有视频/照片中补标 Pose、纸框与文字；如果审查后仍凑不出独立 val/test 父组，再按上表补采。
- 每场采集保留原图/原视频、拍摄参数、摆位草图、真值文件和父组 ID；原件放入 `datasets/raw/`，不直接进训练导出。
- 补采不改变现有 train 配额；val/test 张数仍待后续确定。

## 7. 跨任务复用与相同图片规则

1. **同一任务内**：一个父组只能有一个主 split；其帧、裁剪、掩膜、增强、渲染都随父组走。逐字节完全重复的样本不得跨 train/val/test。
2. **跨任务**：允许复制同一图片到多个任务；每个任务保留自己的标签、类别表、用途和审核状态。父组按任务分别计，不因复制变成新的独立来源。
3. **同图不同模型**：一个图片对 person 是人物目标，对 e_bike 是背景或遮挡物；两份标签分别保存，不能合并成同一训练标签。
4. **同源跨 split**：单父组任务允许 `same_source_allowed_in`，但必须在批次清单和报告中记为同源；不能声称独立场次、独立采集或独立测试。
5. **评估解释**：如果任务 A 的 train 与任务 B 的 test 共享同一父组，B 的测试仍是对 B 模型未见过的像素，但两个任务的结果不能声称互相独立；跨任务比较时说明共享范围。
6. **80% 账本**：按任务×角色统计 train 父组占比；同一父组跨任务使用时，分别计入各任务的账本，但跨任务合并账本要避免同一 Blender 母场景被各任务重复满额。

## 8. 贝叶斯与计划接入

- 每个 split 一份计划：`target_split: train | val | test`；`pairing` 的 `plan_bindings` 指向计划路径，当前 val/test 为空列表。
- 2026-11-15 的 Blender 上调已写入 `config/full_train_allocation.yaml` 的 `source_mix`（fire/bnp/waste_bin 20/40/40，waste_sticker/meter 10/40/50）；编译逐 split 计划时按 split 重新分配，不照搬旧 30/40/30。
- 计划 `strata` 的每一条至少声明：父组 ID、数据源、背景、组合轴状态、标签、配额上下限；分配器只在已声明 strata 内分额度。
- 计划编译前先跑配对校核：父组是否属于该 split、派生是否继承、背景是否已审、同源是否记录、80% 是否满足。
- 当前计划全部是 train（waste_sticker 另有 26 个 test 状态），val/test 预算未定；先只预留父组，不写张数。
- `state_coverage` 仍按计划统计；val/test 计划编译后会分别出现在覆盖表中，不能用 train 的覆盖率代替。
- 跨任务 Blender 账本待实现；同一母场景（arena master、three_buildings、station、waste_bin_pbr）跨任务合计用量要按父场景汇总。

### 8.1 计划对齐与 checker 边界

`source.yaml` 的 `pairing.plan_alignment` 记录现有计划与 pairing split 的冲突；`status: stale` 表示编译前必须重编计划或改 split，不能只改文档假装闭环。当前记录：

- person：已按用户确认把 `occl_009` 改为 train，现有 train `production_plan.json` 已对齐；val/test 仍待 005/006 掩膜或同源预留。
- sign：`bg_ebike_003`（test）、`bg_photo_batch`（val）仍被 train `production_plan.json` 消费。
- waste_sticker：`bg_003`/`bg_005`（val）、`bg_006`（test）仍在 train `production_plan.json` 的背景批次中。
- waste_bin_state：`backgrounds.json` 的 10 train/2 val/2 test 全部同属 `full_scene.mp4`；pairing 只登记 train 主 split 和 `same_source_allowed_in`，val/test 不是独立父组。
- e_bike：组合来源计划给 `bg_car_capture`（val）、`bg_legacy_001`（test）排了 train 份额，尚未重编译为逐 split 计划。

V11（每个 split 的计划只能消费该 split 的父组）目前**没有**由 `source_pairing.py` 自动检查；checker 检查结构、角色枚举、`empty_negative` 的 `confirmed_empty` 条件、capture、路径、来源组成和 `plan_alignment` 结构，计划实际消费内容仍需专项核对。

其他未决：

- 一个 `parent_video` 项含多段视频时，父组数和 80% 上限不能按项计算；当前 e_bike `tr_bike_005_006`、`tr_bike_004_007_010_011`，fire `tr_facade_video_B_001_002` 仍是合并项，编译前需拆开或另记账。
- bnp 把三栋楼整体分到 train/val/test 与“楼面是组合轴”冲突（train 只剩 A 楼）；编译前需按楼内批次重新拆，文档不自行改 split。
- 主检出另一窗口的 `source_ledger` 草图不作为第二套校验器或台账；用户已确认 `source_pairing.py` 为单一 checker，ledger 只作数据访问/实验。

## 9. 实施顺序

1. **确认结构**：本文件的字段、枚举、校核规则和逐任务预留提议。
2. **写入 `source.yaml`（已落盘草案）**：八任务 `pairing` 已按 3.2 添加，`status: draft`；每任务填入父组、split 提议、证据和动作，待结构确认后改为 `reserved`。
3. **扩池审查**：按第 5 节矩阵，用现有审查工具筛查共享池视频在各任务中的背景资格；把 verdict、证据和轴值写回 `pairing.backgrounds`。
4. **补采**：按第 6 节执行 sign、waste_sticker 两场独立采集；meter、bnp 视审查结果决定。
5. **补标**：各任务按第 4 节动作补直接标签、掩膜、桶面、Pose、纸框。
6. **编译逐 split 计划**：先 train，再 val/test；保持 80% 上限、同源记录和跨任务账本。
7. **校核**：运行 `lab/scripts/contracts/source_pairing.py --check`；它校验 pairing 结构、角色枚举、`empty_negative` 的 `confirmed_empty` 条件、ID、`capture_id`、路径、来源组成和 `plan_alignment` 结构。V11 的计划实际消费内容仍待专项核对，任务×split×父组矩阵尚未实现；再跑 `state_coverage`、`full_plan_design` 和目录检查。

## 10. 未决项

- 第 4 节的逐任务 split 提议是否接受；特别是 person 用 012 作 val、CarReal 作 test，fire 按楼面内批次拆，waste_bin 视频 3/2/3。
- 五任务的 20/40/40 与 10/40/50 来源份额已由用户确认采用，不再作为未决项；后续数值调整另行记录。
- `plan_alignment.status: stale` 的 sign/waste_sticker/waste_bin_state/e_bike 需在编译前重编计划或改 split；V11 未自动检查。person 已把 `occl_009` 改为 train。
- bnp 三栋楼整体分 split 与楼面组合轴冲突，需按楼内批次重新拆；e_bike/fire 的多视频合并 parent_video 项需拆父组。
- `source_pairing.py` 为用户确认的单一 checker；`source_ledger` 草图只作数据访问/实验，不作为第二套校验器或台账。
- 生产仍暂停；启动 person 有界批次前需用户明确恢复并同步 `DATASET_PRODUCTION.md` 状态。
- meter、building_name_paper 的补采是否必要；若现有素材补标后能凑出独立父组，可取消。
- 共享池矩阵中 `?` 的背景资格需要逐任务审查后才能确定。
- 跨任务是否需要"全局未见场次"预留；若需要，当前按任务独立分 split 的方案不足。
- `pairing` 的字段名、ID 前缀和 `lab/scripts/contracts/source_pairing.py` 的实现范围。
- val/test 张数何时确定；当前按用户决定只预留父组。
