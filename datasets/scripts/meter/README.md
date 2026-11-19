# 温湿度仪表数据任务

本任务为 `meter`。定位表盘并学习温湿双针对应的八个关键点。 下列描述用于导航，任务目标、标注范围与组合对照以配置中的 `meter` 条目为维护源。

- [学习目标与标注范围](../../../config/task_targets.yaml)；[组合轴与对照](../../../config/combination_axes.yaml)：背景、机距、视角、针位与高光。
- [本任务数据配置](config/dataset.yaml)与[素材来源配置](config/source.yaml)：具体分支读取其支持的字段。
- [制作入口](../README.md#恢复生产入口)列出可执行分支；[任务脚本](./compose_meter_real_background_v4.py)是其中一个入口。产量与接线状态以[当前生产进度](../../../docs/DATASET_PRODUCTION.md#当前生产进度)及实际回执为准。
- [训练入口](../../../lab/features/meter/training/scripts/train_meter_v4.py)及[训练流程](../../../docs/TRAINING_WORKFLOW.md)：按任务配置和实际预算使用；脚本存在不代表模型已训练或完成交付。

数据制作使用同一 [dataset-production-loop](../../../.agents/skills/dataset-production-loop/SKILL.md)，无需按任务复制 Skill。

## 当前全量制作入口

`build_full_dataset.py` 消费 `source.yaml#full_recipe/full_plans` 和共享全量配额设计；当前工作区产物归入 `datasets/exports/meter/{train,val,test}`。执行 `python datasets/scripts/meter/build_full_dataset.py --prepare` 编制三集合图签计划和各自的透明前景渲染计划，再执行 `--pilot` 核对先导；无参数执行全部渲染、合成、审计与导出，或用 `--split train` 单独续作。实际运行使用配方指定的 Blender 可执行文件。

生成器先冻结一份当前工程到 `annotated/meter/full_production/production_inputs/source.blend`，图片资源随工程打包，后台读取副本。透明前景按相机分组分块渲染，完成块有持久回执及产物哈希；合成也按配方的 `composition_chunk_size` 分块审计并保存在 `composites/chunk_*`，复用时核对输入、产物哈希及完成回执；重启从已完成的块接续。val/test 使用专用子帧和焦距变化重新渲染底板，图像、空负标签及放置证据入逐图库后进入合成。共享工程不被保存。

连续双针位置独立采样，网格锚点保留覆盖；两种盘面使用相同几何对照。合成前景在接地区内随机平移与统一缩放，框、八点和显示区四角使用同一个变换。轻微模糊、JPEG 压缩及传感器噪声作用于合成后的 RGB；其参数、原前景几何、变换和可见性随图保存。无表图使用同名空标签文件。

每集合安装前核对图签、八点、负例和 RGB 重复，安装使用可恢复发布。全部完成后生成 `data.yaml`、分集汇总及 V4 逐图来源索引；`--finalize-only` 可单独重建汇总及来源索引，不重做图像。当前工作集不创建正式发布标签或版本目录。小样继续保留在原入口。

## 当前材质与读数配对小样

当前入口为 `build_current_sample.py`；`config/source.yaml` 的 `sample_recipe`、`sample_plan` 和 `sample_manifest` 分别定位配方、已编状态及实际产物。运行 `python datasets/scripts/meter/build_current_sample.py --prepare` 生成有共同刻度几何的白色印刷盘与彩色印刷盘、选源证据和计划，再运行同一入口生成并可恢复地替换当前小样。后台渲染使用 `render_open_sample.py`，不保存共享工程。旧 `render_plan.json`、`production_plan.json` 经 `legacy_render_plan`、`legacy_production_plan` 保留，仅供显式重放，不计入当前计划。

共享底板通过 `backgrounds.py` 查询逐图库当前的 `meter/composite_background` 合格记录，按 `sample_recipe.json#background_selection` 的源 split 与每场景／路线帧数选择；保留当前 `record_id`、父来源、标签和空间证据。公用池及其 extensions 为来源入口，旧候选字段不代替当前资格。实拍底板按已审查接地区放置，空场渲染用近地面图像区域；接地阴影、尺度及透视仍为二维近似。

配方的 `photometry` 将饱和度、整体曝光与前景相对背景亮度分别交叉采样，白盘和彩盘保持同几何、同底板、同光色参数。RGB 变化保留 alpha；每张图保存参数及明度／色度统计。源实拍像素不作这类增强。已有匹配渲染可用 `--reuse-render-from <已保存的原批次目录>` 仅重合成：入口核对渲染状态和关键输入声明，将所复用的 RGBA、掩膜、几何及原清单登记为输入。完整新渲染仍直接运行入口。

全量配额设计维护于 [`full_train_allocation.yaml#tasks.meter`](../../../config/full_train_allocation.yaml)，由 `python lab/scripts/contracts/full_plan_design.py --write` 同步 [三集合设计](../../../docs/FULL_TRAIN_PLAN_DESIGN.md#meter)。它与当前小样计划分别维护；设计配额不计入已产量，val/test 专用渲染及真实底板待补空间的范围在该设计内记录。

两种盘面共同保留原白盘全部黑色刻度像素和同一组端点；温度印刷范围 −30～50 °C、湿度 0～100 %RH 来自现有实拍外观说明，分度仍是参数化合成近似。颜色、数字和文字不包含固定指针。编辑工程中的 `Meter printed white`、`Meter printed colored` 材质可用于同一表盘网格；默认原材质保留，便于旧入口明确回放。首批按 27 机位、独立 6×6 针位及两种材质安排清晰对照，再补局部反射卡对照；实际图数、组合数、逐点可见性和高光像素变化以小样 `checks.json` 为准。Cycles 禁用会抹掉细小刻度的去噪，使用 64 samples 和 Box 像素滤波。

输出包括 `foreground/` 的透明 RGBA、`images/` 的中灰底渲染预览及真实底图合成、对应 `labels/` 的 29 列 YOLO Pose、`annotations/`、八点叠加 `overlays/` 与表盘放大 `crops/`。仅用 `contact_sheet.jpg` 一张图片汇总实拍、点位放大、渲染和两类底图合成，不生成 HTML。关键点序号依次为温轴、温尖、湿轴、湿尖、温最小、温最大、湿最小、湿最大。检查父对象和局部坐标、独立针角、投影针尖、材质间几何不变，以及合成后的统一坐标变换；v1 表示已知合成几何但弱可辨或遮挡，不能当成可读保证。

本批真实底图选自同一 `full_scene.mp4` 的两个已完整确认仪表空负的 train 帧，并按本次原图目视核对中心放置区；保留逐图库记录 ID、图像哈希及父来源。640×360 底图等比裁成 640×480，前景与框点同移，接触阴影是明确记录的二维近似。小样保持 `unassigned`，保留父来源 train 接触，不是正式版本或独立评估集。替换前旧 12 张小样保存在 `Archive/technical/datasets/meter/sample_before_material_variants_20261118/`。

## Blender 编辑场景

当前混合 sample 接入 55 张任务实拍，逐图直接视觉标注保存在 `datasets/annotated/meter/real_meter_visual_manifest.jsonl` 的 `keypoints`，每图 JSON 与实心点预览在 `datasets/annotated/meter/real_pose_visual/`。54 张八点齐全的实拍经 `direct_task_pose` 进入 `images/`、`labels/`；最后一张出画图保存 3 个可辨点，其余为 null，经 `direct_task_review` 保留，不编造完整 Pose。实拍保持原像素、原用途历史与 native 尺寸，YOLO 坐标按各图实际宽高归一化。所有 Pose 预览改为实心点及点号，保留表框，不用空心圈覆盖针尖。当前共 307 张：54 实拍 Pose、1 部分点实拍、126 Blender 预览、52 真实底图合成、74 Blender 底图合成，共 306 份 Pose 标签，另有 126 份透明前景；原标注和原 sample 已分别留快照。图签仍为 unassigned，AI 视觉标注不等于独立实物读数标定。

统一工程内只维护 `Meter Station` 一套站房和双针表盘。`Station dial tilt pivot` 控制表盘安装倾角，两针继续由各自的旋转对象控制。2026-11-18 移除了独立复制的 `Meter Station Tilt` 场景；它的旧贴图和端点与现行参数化表盘不同，不能混作同一份真值。

旧站房以 Y 轴朝上、Z 轴朝相机，普通透视视图会显示整个房间侧倒。现用 `Meter Station | Editor Z-up frame` 将整个站房、房间、灯光和相机一起旋转到 Z 轴朝上并缩放至米制；灯光功率同步按缩放平方调整。该父对象下保留原有局部坐标、八点与电表 ROI。两个生产入口在独立后台副本中调用 `restore_station_production_frame`，恢复已有计划使用的局部单位和灯光功率；不将编辑父变换再次叠到 Arena 装配上。

修改前工程及 `.blend1` 保存在 `Archive/technical/revisions/meter/scene_before_upright_20261118/`。投影对照、修复记录和正面及近景渲染见 [修复证据](../../annotated/meter/station_workspace_repair_20261118/repair.json)。已在 Blender 5.2.3 LTS 后台核对原局部坐标下八点及 ROI 投影完全一致，并通过场地和独立站房的实际预检入口；这次修复不改变现有小样的生成时记录。

当前编辑场景采用开放前景布置，默认 PNG RGBA、透明世界。旧房间、地板收在 `Room reference (disabled)`，视口和渲染均关闭；灯光与相机位于 `Open lights and cameras`。室内环境贴图和三盏 Area 灯仍用于照明及玻璃反射，透明背景不等于关闭环境照明。世界旋转使用 Z-up 方向，后台入口恢复原站房坐标时同步转换环境方向。开放布置前的完整内存快照位于 `Archive/technical/revisions/meter/before_open_stage_20261118/aiaec_dataset.blend`。

**相机预设与调整**：参考 `Waste Stickers` 的左／中／右 × 远／中／近九机位结构，本任务独立使用方位角 −15°／0°／15°、距离 2.50／1.90／1.35 米、焦距 35 mm；每个机位另有仰视 8°、平视及俯视 8°，共 27 台预设相机，保存在 `Meter Station | Orbit cameras`。相机名后缀 `up8` 表示相机低于瞄准点、向上看；`down8` 表示相机高于瞄准点、向下看。`elevation_deg` 为相机位置的仰角，因此 `up8=-8`、`down8=+8`。瞄准点为整套表盘与支座的包围框中心，安装倾角仍由表盘自己的 pivot 控制。

`Meter Station | Camera orbit controls` 的自定义属性控制默认 `Preview vehicle front` 相机：`distance_m`、`yaw_deg`、`elevation_deg`、`target_x_m`、`target_y_m`、`target_height_m` 与 `focal_length_mm`。这台可调相机与 27 台固定预设分开。后台入口先求值并烘焙可调相机，再移除编辑控制器及其驱动，使已有批次计划能够直接设置局部相机参数。固定预设是当前工程和对照图的机位库存，现行旧批次计划仍按其自身参数执行。

27 个预设均已渲染透明对照，完整仪表和支座均在画幅内。相机参数、RGBA 检查、图签投影与两入口预检记录见 [开放场景证据](../../annotated/meter/open_stage_20261118/repair.json)，[27 机位总览](../../annotated/meter/open_stage_20261118/contact_sheet.jpg)。这批为场景核对图，不计作新的正式训练导出。

现有实景合成计划读取原审查清单第 4、5、6 行（`frame_003601`、`frame_009901`、`frame_010501`）。这些 `images` 行由入口通过 `review_record` 和原清单位置追溯，媒体查询仍使用现有索引。选用其他底图时应按仪表任务核对表盘放置区域与图签证据。

[实景合成入口](compose_meter_real_background_v4.py)可直接复用相同已完成批次。需要重做时给同一 `--output` 加 `--replace`，入口按旧清单和生产回执只替换该合成批次的生成文件，保留其他文件。旧清单或回执不符、产物名与人工文件或另一批次冲突时会报错。共享 Arena 底图的资格与本入口要求的固定投影属于不同用途证据，按 `config/source.yaml` 的实际选源和逐图库记录核对。

<!-- TASK_DESIGN:START -->

## 学习目标与组合设计

本节由共享配置生成；修改目标与组合请编辑 [task_targets.yaml](../../../config/task_targets.yaml) 和 [combination_axes.yaml](../../../config/combination_axes.yaml)，运行 `python lab/scripts/contracts/state_coverage.py --write`。

**学习及标注**

- 类别：meter_dial
- 关键点：8
- 指针：temperature、humidity
- 有无：single、absent
- 负例：absent

可辨表盘及八点分别判定；遮挡或高光下逐点记录可见性，未知点按 Pose 入口的监督资格选择用途。

温针和湿针各含轴心、针尖和量程两端，针位为目标取值；物理读数由刻度映射得到。

**素材用途**

- `direct_target`：目标表盘框和入口可消费的八点真值及可见性齐全。
- `target_foreground`：有坐标映射、点位和掩膜的双针表盘图或渲染，保留温湿针位记录。
- `hard_negative`：确认无目标表盘的相似外形物体。

**组合轴与对照**

以下是本任务的设计条件；`dynamic` 表示依据实际素材和任务计划确定。

- `background`：dynamic
- `distance`：dynamic
- `view`：frontal、oblique
- `glare`：none、present
- `face_material`：white、colored
- `reading_sampling`：grid_anchor、continuous
- `placement`：fixed_contact、randomized
- `degradation`：clean、soft_blur、jpeg、sensor_noise、blur_jpeg
- `saturation`：native、low、very_low
- `exposure`：dark、normal、bright
- `light_relation`：foreground_darker、matched、foreground_brighter

**布局**：站房支撑位置或表盘替换区及坐标尺寸；针位、机位和反光依各自证据记录。

**material_background_photometry**：白盘和彩盘不能与背景、饱和度或明暗关系绑定。

- 安排：同机位同双针同底图成对白盘和彩盘；独立轮换饱和度、整体曝光及前景相对背景亮度。
- 核对：从 parameters 和 photometry 联合统计三轴，逐点坐标随统一缩放平移；RGB 增强保留 alpha，逐点记录可见性。

**reading_view_light**：表针读数不能从机距、俯视程度或光色猜出。

- 安排：同一针位保持安装倾角，分别换机距与入射角；同机位换针位并独立轮换色调。
- 核对：用相机到盘心距离、表盘法向入射角及针位作联合表并核对 8 点标签；ground_angle_deg 只记录安装倾角。

**background_glare**：背景和局部高光不能决定是否有表或某个点是否可见。

- 安排：同背景有表／无表，另选相似圆形物作硬负；同读数保持机位配局部高光与无高光。
- 核对：核对框与各点可见性并回链 foreground_annotation；高光、读数和安装姿态都应有各自证据。

对照描述来自设计配置；结构化记录目前为部分编码，具体制作按任务入口和可用素材安排。

<!-- TASK_DESIGN:END -->
