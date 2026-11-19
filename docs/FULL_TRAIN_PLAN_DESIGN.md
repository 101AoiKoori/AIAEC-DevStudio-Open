# 八任务全量 train 计划设计（草案）

> 状态：`design_draft`。本文由 `config/full_train_allocation.yaml` 生成，不是已执行计划，也不覆盖各任务当前的 `production_plan.json`。组配额是设计分配，不是实际产量、实测分布或模型效果。

## 设计原则

- 训练目标（类别、关键点、OCR、计数）与组合轴（光照、视角、遮挡、背景等噪声）分开；目标见 `config/task_targets.yaml`，轴见 `config/combination_axes.yaml`。
- 对象分类：本文件是 GeneratedDoc，由 PlanDraft `config/full_train_allocation.yaml` 生成；PlanDraft 不是 ProductionPlan，不被生产守卫消费。编译后的 `datasets/scripts/<任务>/config/production_plan.json` 才是 ProductionPlan。
- 组配额只安排已声明的可行组合；不要求每个组合都有独立实拍对应证据。
- 来源组成和 Blender 参考份额读取 `source_mix` 和 `blender_cap`；标为历史参考的比例不规定现行编批。按可用直接实拍和可行合成编批，实际份额按批次清单记录。
- 80% 参考份额用于提示父组集中度。编批时分配已合格的库存。同源 val/test 记录诊断范围，独立实拍评估另核物理父组与场次。底图容量和成图配额分别计数。
- 由各任务现用计划入口编制具体状态。生成、训练和评估各自使用对应入口。

## 总览

| 任务 | train 目标（张） | 现有 train 计划预算（张） | 现有计划容量（strata max_count 合计，张） | 需要新增容量（张） | 直接实拍 | 实拍合成 | Blender |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| person | 3000 | 2992 | 2992 | 8 | 30% | 40% | 30% |
| e_bike | 4000 | 0 | 0 | 4000 | 30% | 40% | 30% |
| fire | 3000 | 3000 | 3000 | 0 | 0.7% | 59.3% | 40% |
| building_name_paper | 3000 | 3000 | 3000 | 0 | 按实际编批 | 按实际编批 | 按实际编批 |
| sign | 2500 | 2500 | 2500 | 0 | 0.36% | 67.64% | 32% |
| waste_bin_state | 3500 | 0 | 0 | 3500 | 0.8% | 22.86% | 76.34% |
| waste_sticker | 1800 | 1800 | 1800 | 0 | 按实际编批 | 按实际编批 | 按实际编批 |
| meter | 6480 | 6480 | 6480 | 0 | 按实际编批 | 按实际编批 | 按实际编批 |

预算、计划容量和需要新增容量均以最终训练图片张数计。计划容量为数据集用途的 train 计划 strata 的 `max_count` 之和，源渲染和小样另见组合轴覆盖表。父组数见各任务计划与 `docs/PARENT_SOURCE_CAPACITY.md`。

## 分任务设计

### person

- 训练目标：classes: community, noncommunity；counts: empty, single, multi
- 全量目标：3000 张；现有 train 计划预算 2992 张；现有计划容量 2992 张；需要新增容量 8 张。
- 来源组成参考为直接实拍 30%、实拍背景合成 40%、Blender 30%，Blender 份额提示线为 40%。
- 使用已导出的 3,000／300／300 批次作基线。后续按实际失败选单人、仅白背、正背同框及两类人物的无遮挡／车挡对照，优先复用合格底图换身份与有效脚点。
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
- 全量目标：4000 张；现有 train 计划预算 0 张；现有计划容量 0 张；需要新增容量 4000 张。
- 来源组成参考为直接实拍 30%、实拍背景合成 40%、Blender 30%，Blender 份额提示线为 40%。
- 当前子批预算与状态从电动车 production_plan.json 读取，实际图签从当前小样清单读取。追加批次先在合格底图上换脚点、视面和尺度并交叉两姿态，再单列能通过布局核对的多车和遮挡状态。
- 设计说明：直立与倒伏正例各 1,800；零／单／多车约 400／1,600／2,000。街道和停车区各半是抽样参考，先安排已有可行格。

| 组 | 目标 | 组合轴 | 配额 |
| --- | --- | --- | ---: |
| empty_street | empty | count=empty, location=street | 200 |
| empty_parking | empty | count=empty, location=parking | 200 |
| upright_single_street | upright | count=single, location=street | 400 |
| upright_single_parking | upright | count=single, location=parking | 400 |
| upright_multi_street | upright | count=multi, location=street | 500 |
| upright_multi_parking | upright | count=multi, location=parking | 500 |
| fallen_single_street | fallen | count=single, location=street | 400 |
| fallen_single_parking | fallen | count=single, location=parking | 400 |
| fallen_multi_street | fallen | count=multi, location=street | 500 |
| fallen_multi_parking | fallen | count=multi, location=parking | 500 |

### fire

- 训练目标：classes: fire_sticker；counts: zero, single, multi
- 全量目标：3000 张；现有 train 计划预算 3000 张；现有计划容量 3000 张；需要新增容量 0 张。
- 来源组成参考为直接实拍 0.7%、实拍背景合成 59.3%、Blender 40%，Blender 份额提示线为 40%。
- 当前四种生成方式小样已生成并核对；实际数量与逐图归属见 datasets/samples/fire/manifest.json，全量分配由 datasets/scripts/fire/config/full_dataset_plan.json 维护。
- 设计说明：零/单/多贴约 600/1,200/1,200；各楼面组为设计参考。当前按真实父来源及可行墙面执行，Blender 配额包含实景混合合成，实际逐楼计数以可见实例归属为准。

| 组 | 目标 | 组合轴 | 配额 |
| --- | --- | --- | ---: |
| facade_A_zero | empty | count=zero, facade=A | 200 |
| facade_A_single | fire_sticker | count=single, facade=A | 400 |
| facade_A_multi | fire_sticker | count=multi, facade=A | 400 |
| facade_B_zero | empty | count=zero, facade=B | 200 |
| facade_B_single | fire_sticker | count=single, facade=B | 400 |
| facade_B_multi | fire_sticker | count=multi, facade=B | 400 |
| facade_C_zero | empty | count=zero, facade=C | 200 |
| facade_C_single | fire_sticker | count=single, facade=C | 400 |
| facade_C_multi | fire_sticker | count=multi, facade=C | 400 |

### building_name_paper

- 训练目标：classes: paper；counts: zero, single, multi；ocr: text
- 全量目标：3000 张；现有 train 计划预算 3000 张；现有计划容量 3000 张；需要新增容量 0 张。
- 来源组成按当批可用的直接实拍目标和合成确定；旧的直接实拍 20%、实拍合成 40%、Blender 40% 及 Blender 40% 提示线仅供历史参考，不约束现行编批。
- 已完成 train 3000、val 300、test 300 的当前全量图签。实际分配读取 allocation_recipe，逐图数量与检测框读取 exports/building_name_paper/audit.json；保留四分支小样及原始证据。
- 设计说明：下列 source_mix 和 groups 保留早期设计参考，不作为当前执行配额；当前按 full_recipe 编排七种楼数组合及全部 26 种纸条开关状态，监督为 paper 检测。

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
- 全量目标：2500 张；现有 train 计划预算 2500 张；现有计划容量 2500 张；需要新增容量 0 张。
- 来源组成参考为直接实拍 0.36%、实拍背景合成 67.64%、Blender 32%，Blender 份额提示线为 40%。
- 已生成Train2500、Val300、Test300；当前产量、图签和覆盖读取datasets/exports/sign/manifest.json与datasets/annotated/sign/full_dataset_audit.json。保留440张已接受小样作为对照，背景父来源分集，前景同源。
- 设计说明：Train2500、Val300、Test300按任务full_dataset配方生成。目标覆盖完整、截断/遮挡、杆底座及普通杆对照；实拍按现货加入，前景共享同一牌具视频，Val为受控诊断，Test停车视频已有历史接触，作回归。

| 组 | 目标 | 组合轴 | 配额 |
| --- | --- | --- | ---: |
| complete | sign_fixture | truncation=none | 1300 |
| partial_or_occluded | sign_fixture | - | 500 |
| identifiable_pole | sign_fixture | truncation=pole_only | 200 |
| absent | no_sign | - | 300 |
| ordinary_pole | ordinary_pole | - | 200 |

### waste_bin_state

- 训练目标：classes: open, closed；counts: empty, single, multi
- 全量目标：3500 张；现有 train 计划预算 0 张；现有计划容量 0 张；需要新增容量 3500 张。
- 来源组成参考为直接实拍 0.8%、实拍背景合成 22.86%、Blender 76.34%，Blender 份额提示线为 76%。
- 以已导出的 3,500 张 train 和 val／test 子批为起点。补量优先同底图同桶色换开闭、交换桶位及邻桶遮挡，共用 Blender 母场景的评估按同源诊断记账。
- 设计说明：直拍 28（4 正、24 空），实拍底图合成 800，Blender 轨道底图合成 2396 与空负 276。空场 300、非空 3200；单/多桶各半，合成在同源底图上按颜色配对开闭。轨道底图共享一个 Blender 母场景，独立来源容量按实际父组统计。

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

- 训练目标：classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23；counts: empty, single, multi；presence: present, absent
- 全量目标：1800 张；现有 train 计划预算 1800 张；现有计划容量 1800 张；需要新增容量 0 张。
- 来源组成按当批可用的直接实拍目标和合成确定；旧的直接实拍 0%、实拍合成 30%、Blender 70% 及 Blender 70% 提示线仅供历史参考，不约束现行编批。
- 已完成 Train 1800、Val 240、Test 240 张，共 5130 个框，audit passed。每图 1–4 个可见桶，四路来源按实际清单记录。后续以当前完整数据训练基线，共享固定图案与 Blender 基础资产的评估范围为同源诊断。
- 设计说明：Train 的 1,800 张规划含 4,050 个有效贴纸标签，24 类平均约 169 个。Val 和 Test 各 240 张及 540 个有效标签，每类平均 22.5 个，完整四路来源、分集配额及回归范围见 [垃圾贴纸数据设计](DATASET_DESIGN.md#垃圾贴纸数据设计)。

| 组 | 目标 | 组合轴 | 配额 |
| --- | --- | --- | ---: |
| no_sticker | absent | visible_sticker_count=0 | 180 |
| one_sticker | present | visible_sticker_count=1 | 405 |
| two_stickers | present | visible_sticker_count=2 | 405 |
| three_stickers | present | visible_sticker_count=3 | 405 |
| four_stickers | present | visible_sticker_count=4 | 405 |

### meter

- 训练目标：classes: meter_dial；presence: single, absent；keypoints: 8；needle_axes: temperature, humidity
- 全量目标：6480 张；现有 train 计划预算 6480 张；现有计划容量 6480 张；需要新增容量 0 张。
- 来源组成按当批可用的直接实拍目标和合成确定；旧的直接实拍 10%、实拍合成 40%、Blender 50% 及 Blender 50% 提示线仅供历史参考，不约束现行编批。
- 共享逐图库按 meter/composite_background 资格编制；3 场景×8 路线分层选底板，白盘和彩盘成对，独立轮换饱和度、曝光、前景相对亮度。
- 设计说明：首轮 Train 6480 / Val 1080 / Test 1080，合计 8640；负例各占 10%。现有 54 张实拍按 45/7/2 父组划分。3 个 train 实拍底板重复合成保留父源复用统计；不按旧 10/40/50 比例复制实拍。全量已生成并通过图签与 V4 数据预检；实际产物见 datasets/exports/meter/manifest.json。

**三集合配额（当前工作集已生成）**

| 集合 | 总数 | 原生实拍 | 实拍底板合成 | Blender 底板合成 | 无表负例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 6480 | 45 | 648 | 5139 | 648 |
| val | 1080 | 7 | 0 | 965 | 108 |
| test | 1080 | 2 | 0 | 970 | 108 |

2 材质×27 机位×36 独立双针组合=1944 基本组合；train 正例 5832 约为三轮背景/成像变化，加入 648 无表图。原生实拍占用相应正例配额。

以下比例作用于生成正例；原生实拍保留真实图像并单独统计，不强行更改状态。

- face_material：white 50%、colored 50%
- saturation：native 40%、low 40%、very_low 20%
- exposure：normal 50%、dark 25%、bright 25%
- light_relation：matched 50%、foreground_darker 25%、foreground_brighter 25%
- distance：near_1p35m 35%、middle_1p90m 40%、far_2p50m 25%
- elevation：up8 20%、level 60%、down8 20%
- yaw：left15 30%、frontal 40%、right15 30%
- glare：none 75%、present 25%
- occlusion：none 90%、local 10%

**来源和评估范围**

- authority：逐图库 meter/composite_background；共用池及 extensions 用于发现，不升级旧 candidate 字段。
- train：现有 576 张 Blender 底板及 3 张实拍接地区；完整源池均可选，小样分层取 96+3。
- val：待补放置区域的实拍分支本轮回退为专用 Blender 底板；val 432 张份额并入 Blender，合计 965 张。
- test：生成 test 专用背景渲染；不同路径、相机参数、种子和布局。现有 train 底板不复制进 test。
- shared_asset_scope：3 灯光场景共享同一个 .blend 父资产；val/test 新渲染也属于同资产参数泛化诊断。
- paired_controls：同几何/双针/底图/光色的白盘和彩盘成组；高光/无高光成组，不跨 split；有表与无表按底板成组。
- negative_content：10% 无表；优先任务已证实空负。圆形灯、轮子、图标等有证据时作为无表难例，不能仅按形状自动判空。

**后续编批项**

- val 的 20 张实拍目前仅有完整空负标签，合成须补接地区域；该分支未就绪时可在同总额内以 val 专用 Blender 底板补足并记实际来源。
- val/test 专用前景与底板已独立渲染；未复制 train 图像进入 val/test。
- 全量已执行局部反光、低饱和、明暗、随机位置尺度、连续双针与轻微成像退化；局部遮挡使用有记录的二维边缘掩膜。

本文件为配额设计，不是生产计划或已导出数据；先执行可行 train，再依据基线失败补数据，不因独立实拍不足停止。

| 组 | 目标 | 组合轴 | 配额 |
| --- | --- | --- | ---: |
| direct_real | meter_dial | source=direct_visual_pose | 45 |
| real_background_composite | meter_dial | source=real_background | 648 |
| blender_background_composite | meter_dial | source=shared_rendered_background | 5139 |
| empty_negative | absent | presence=absent | 648 |

## 编制可执行批次

1. 查询已有资格的共享底图、Blender 或实拍来源，纳入当批可行状态并分配预算。定向补判仍缺证据的用途。
2. 在计划中分配 split 并记录父组用途和已知接触。同一母场景或拍摄原件按实际谱系计数。合格底图可承载多个有效布局，按最终图签计产量。
3. 把组配额展开为 `production_plan.json` 的 strata，写清背景、目标、标签、增强、参数、来源和上下限。
4. 按出图清单登记来源、覆盖和未补容量，按实际状态调整当批份额。设计或参考计划变更后同步组合轴覆盖文档。
5. 当前可运行的批次见 [生产进度](DATASET_PRODUCTION.md#当前生产进度)。
