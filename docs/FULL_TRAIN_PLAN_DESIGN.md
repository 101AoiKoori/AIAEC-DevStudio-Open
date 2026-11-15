# 八任务全量 train 计划设计（草案）

> 状态：`design_draft`。本文由 `config/full_train_allocation.yaml` 生成，不是已执行计划，也不覆盖各任务当前的 `production_plan.json`。组配额是设计分配，不是实际产量、实测分布或模型效果。

## 设计原则

- 训练目标（类别、关键点、OCR、计数）与组合轴（光照、视角、遮挡、背景等噪声）分开；目标见 `config/task_targets.yaml`，轴见 `config/combination_axes.yaml`。
- 对象分类：本文件是 GeneratedDoc，由 PlanDraft `config/full_train_allocation.yaml` 生成；PlanDraft 不是 ProductionPlan，不被生产守卫消费。编译后的 `datasets/scripts/<任务>/config/production_plan.json` 才是 ProductionPlan。
- 组配额只安排已声明的可行组合；不要求每个组合都有独立实拍对应证据。
- 来源组成按各任务 `source_mix`：person/e_bike/sign 为 30/40/30；fire/building_name_paper/waste_bin_state 为 20/40/40；waste_sticker/meter 为 10/40/50。Blender 上限按任务声明；份额是设计基准，不是实测分布。
- 每个来源域在 train 最多使用其合格父组的 80%，其余留给 val/test 与后续补量；编译前须核对来源容量和父组上限。
- 本草案不启动生产、训练或验证；编译成可执行计划前需用户确认。

## 总览

| 任务 | train 目标（张） | 现有 train 计划预算（张） | 现有计划容量（strata max_count 合计，张） | 需要新增容量（张） | 直接实拍 | 实拍合成 | Blender |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| person | 3000 | 36 | 54 | 2946 | 30% | 40% | 30% |
| e_bike | 4000 | 48 | 72 | 3928 | 30% | 40% | 30% |
| fire | 3000 | 18 | 108 | 2892 | 20% | 40% | 40% |
| building_name_paper | 3000 | 986 | 1026 | 1974 | 20% | 40% | 40% |
| sign | 2500 | 400 | 400 | 2100 | 30% | 40% | 30% |
| waste_bin_state | 3500 | 19 | 19 | 3481 | 20% | 40% | 40% |
| waste_sticker | 7200 | 193 | 193 | 7007 | 10% | 40% | 50% |
| meter | 3500 | 48 | 48 | 3452 | 10% | 40% | 50% |

预算、计划容量和需要新增容量均以最终训练图片张数计；计划容量是现有计划 strata 的 `max_count` 之和，不是父源容量。渲染状态、组合状态和父组数分别见各任务计划与 `docs/PARENT_SOURCE_CAPACITY.md`。

## 分任务设计

### person

- 训练目标：classes: community, noncommunity；counts: empty, single, multi
- 全量目标：3000 张；现有 train 计划预算 36 张；现有计划容量 54 张；需要新增容量 2946 张。
- 来源组成：直接实拍 30%、实拍背景合成 40%、Blender 30%；Blender 上限 40%。
- 设计说明：空场 300；非空 2,700 中 single/multi 各约 1,350；电动车遮挡共 750，直立/倒伏各 375，两类身份各约一半。

| 组 | 目标 | 组合轴 | 配额 |
| --- | --- | --- | ---: |
| empty_scene | empty | vehicle_occlusion=none | 300 |
| community_none | community | vehicle_occlusion=none | 975 |
| noncommunity_none | noncommunity | vehicle_occlusion=none | 975 |
| community_upright | community | vehicle_occlusion=upright | 188 |
| noncommunity_upright | noncommunity | vehicle_occlusion=upright | 187 |
| community_fallen | community | vehicle_occlusion=fallen | 187 |
| noncommunity_fallen | noncommunity | vehicle_occlusion=fallen | 188 |

### e_bike

- 训练目标：classes: upright, fallen；counts: empty, single, multi
- 全量目标：4000 张；现有 train 计划预算 48 张；现有计划容量 72 张；需要新增容量 3928 张。
- 来源组成：直接实拍 30%、实拍背景合成 40%、Blender 30%；Blender 上限 40%。
- 设计说明：直立/倒伏各 2,000；零/单/多车约 400/1,600/2,000；街道/停车区各半；人车和车车遮挡在组内轮换，截断约 500；零车组 target=empty，不带姿态类别。

| 组 | 目标 | 组合轴 | 配额 |
| --- | --- | --- | ---: |
| upright_empty_street | empty | count=empty, location=street | 100 |
| upright_empty_parking | empty | count=empty, location=parking | 100 |
| upright_single_street | upright | count=single, location=street | 400 |
| upright_single_parking | upright | count=single, location=parking | 400 |
| upright_multi_street | upright | count=multi, location=street | 500 |
| upright_multi_parking | upright | count=multi, location=parking | 500 |
| fallen_empty_street | empty | count=empty, location=street | 100 |
| fallen_empty_parking | empty | count=empty, location=parking | 100 |
| fallen_single_street | fallen | count=single, location=street | 400 |
| fallen_single_parking | fallen | count=single, location=parking | 400 |
| fallen_multi_street | fallen | count=multi, location=street | 500 |
| fallen_multi_parking | fallen | count=multi, location=parking | 500 |

### fire

- 训练目标：classes: fire_sticker；counts: zero, single, multi
- 全量目标：3000 张；现有 train 计划预算 18 张；现有计划容量 108 张；需要新增容量 2892 张。
- 来源组成：直接实拍 20%、实拍背景合成 40%、Blender 40%；Blender 上限 40%。
- 设计说明：零/单/多贴约 600/1,200/1,200；三个楼面各 1,000；贴放位置、尺度、偏航、遮挡和成像在组内轮换。

| 组 | 目标 | 组合轴 | 配额 |
| --- | --- | --- | ---: |
| facade_A_zero | fire_sticker | count=zero, facade=A | 200 |
| facade_A_single | fire_sticker | count=single, facade=A | 400 |
| facade_A_multi | fire_sticker | count=multi, facade=A | 400 |
| facade_B_zero | fire_sticker | count=zero, facade=B | 200 |
| facade_B_single | fire_sticker | count=single, facade=B | 400 |
| facade_B_multi | fire_sticker | count=multi, facade=B | 400 |
| facade_C_zero | fire_sticker | count=zero, facade=C | 200 |
| facade_C_single | fire_sticker | count=single, facade=C | 400 |
| facade_C_multi | fire_sticker | count=multi, facade=C | 400 |

### building_name_paper

- 训练目标：classes: paper；counts: zero, single, multi；ocr: text
- 全量目标：3000 张；现有 train 计划预算 986 张；现有计划容量 1026 张；需要新增容量 1974 张。
- 来源组成：直接实拍 20%、实拍背景合成 40%、Blender 40%；Blender 上限 40%。
- 设计说明：无纸 1,000、有纸 2,000；有纸中单名 1,200、多名 800；单楼/多楼各 1,500；OCR 文字由生成字符串或已核记录提供。

| 组 | 目标 | 组合轴 | 配额 |
| --- | --- | --- | ---: |
| no_paper_single_building | no_paper | building_count=single | 500 |
| no_paper_multi_building | no_paper | building_count=multi | 500 |
| paper_single_name_single_building | paper | building_count=single, name=single | 600 |
| paper_single_name_multi_building | paper | building_count=multi, name=single | 600 |
| paper_multiple_names_single_building | paper | building_count=single, name=multiple | 400 |
| paper_multiple_names_multi_building | paper | building_count=multi, name=multiple | 400 |

### sign

- 训练目标：classes: sign_fixture；presence: placed, absent
- 全量目标：2500 张；现有 train 计划预算 400 张；现有计划容量 400 张；需要新增容量 2100 张。
- 来源组成：直接实拍 30%、实拍背景合成 40%、Blender 30%；Blender 上限 40%。
- 设计说明：有牌 2,000、无牌 300、普通立杆 200；有牌中截断 500（仅可辨牌杆约 100），正视面为主，侧背视面保留少量。

| 组 | 目标 | 组合轴 | 配额 |
| --- | --- | --- | ---: |
| placed_front_none | sign_fixture | truncation=none, view=front | 750 |
| placed_front_partial | sign_fixture | truncation=partial, view=front | 250 |
| placed_oblique_none | sign_fixture | truncation=none, view=oblique | 500 |
| placed_oblique_partial | sign_fixture | truncation=partial, view=oblique | 200 |
| placed_side_none | sign_fixture | truncation=none, view=side | 150 |
| placed_side_partial | sign_fixture | truncation=partial, view=side | 50 |
| placed_back_none | sign_fixture | truncation=none, view=back | 100 |
| absent | no_sign | - | 300 |
| ordinary_pole | ordinary_pole | - | 200 |

### waste_bin_state

- 训练目标：classes: open, closed；counts: empty, single, multi
- 全量目标：3500 张；现有 train 计划预算 19 张；现有计划容量 19 张；需要新增容量 3481 张。
- 来源组成：直接实拍 20%、实拍背景合成 40%、Blender 40%；Blender 上限 40%。
- 设计说明：空场 300；非空 3,200；四色各 800，开/闭各半，单/多桶各半；低视角、邻桶遮挡和成像在组内轮换。

| 组 | 目标 | 组合轴 | 配额 |
| --- | --- | --- | ---: |
| empty | empty | - | 300 |
| blue_open_single | open | color=blue, count=single | 200 |
| blue_open_multi | open | color=blue, count=multi | 200 |
| blue_closed_single | closed | color=blue, count=single | 200 |
| blue_closed_multi | closed | color=blue, count=multi | 200 |
| red_open_single | open | color=red, count=single | 200 |
| red_open_multi | open | color=red, count=multi | 200 |
| red_closed_single | closed | color=red, count=single | 200 |
| red_closed_multi | closed | color=red, count=multi | 200 |
| green_open_single | open | color=green, count=single | 200 |
| green_open_multi | open | color=green, count=multi | 200 |
| green_closed_single | closed | color=green, count=single | 200 |
| green_closed_multi | closed | color=green, count=multi | 200 |
| gray_open_single | open | color=gray, count=single | 200 |
| gray_open_multi | open | color=gray, count=multi | 200 |
| gray_closed_single | closed | color=gray, count=single | 200 |
| gray_closed_multi | closed | color=gray, count=multi | 200 |

### waste_sticker

- 训练目标：classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23；counts: single, multi
- 全量目标：7200 张；现有 train 计划预算 193 张；现有计划容量 193 张；需要新增容量 7007 张。
- 来源组成：直接实拍 10%、实拍背景合成 40%、Blender 50%；Blender 上限 50%。
- 设计说明：24 类 × 4 桶色 × 75 张 = 7,200；每类 300 张图、300 个实例，每色 1,800 张；默认单张单贴，后续可加多贴但不抵扣当前每类配额。

| 组 | 配额 |
| --- | ---: |
| class 0–23 × colors red, blue, green, gray × 每格 75 | 7200 |

### meter

- 训练目标：classes: meter_dial；presence: single, absent；keypoints: 8；needle_axes: temperature, humidity
- 全量目标：3500 张；现有 train 计划预算 48 张；现有计划容量 48 张；需要新增容量 3452 张。
- 来源组成：直接实拍 10%、实拍背景合成 40%、Blender 50%；Blender 上限 50%。
- 设计说明：12 个针位各约 291–292 张；反光约 875 张（25%）；距离、视角、背景和成像在组内轮换。

| 组 | 目标 | 组合轴 | 配额 |
| --- | --- | --- | ---: |
| needle_0.1_0.25 | meter_dial | needle=0.1:0.25 | 292 |
| needle_0.1_0.75 | meter_dial | needle=0.1:0.75 | 292 |
| needle_0.26_0.25 | meter_dial | needle=0.26:0.25 | 292 |
| needle_0.26_0.75 | meter_dial | needle=0.26:0.75 | 292 |
| needle_0.42_0.25 | meter_dial | needle=0.42:0.25 | 292 |
| needle_0.42_0.75 | meter_dial | needle=0.42:0.75 | 292 |
| needle_0.58_0.25 | meter_dial | needle=0.58:0.25 | 292 |
| needle_0.58_0.75 | meter_dial | needle=0.58:0.75 | 292 |
| needle_0.74_0.25 | meter_dial | needle=0.74:0.25 | 291 |
| needle_0.74_0.75 | meter_dial | needle=0.74:0.75 | 291 |
| needle_0.9_0.25 | meter_dial | needle=0.9:0.25 | 291 |
| needle_0.9_0.75 | meter_dial | needle=0.9:0.75 | 291 |

## 编译到可执行计划的前置条件

1. 核对各任务 `source.yaml`、清单和实际文件，确认组配额能由合格父来源承载；不足时调整受影响组。
2. 按父组分配 80% 上限，保留 val/test 与后续补量；同一母场景/原件只计一个父组。
3. 把组配额展开为 `production_plan.json` 的 strata，写清背景、目标、标签、增强、参数、来源和上下限。
4. 编译后运行 `python lab/scripts/contracts/state_coverage.py --write` 与 `--check`，确认目标/轴覆盖和容量缺口。
5. 本草案与当前首批计划分开；首批执行和扩量顺序按 `docs/DATASET_PRODUCTION.md`，不因草案跳过首批。
