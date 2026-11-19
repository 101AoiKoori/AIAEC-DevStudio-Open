# 楼名纸条数据任务

本任务为 `building_name_paper`。定位楼名纸片；文字可读且真值已知时另记录 OCR 文字。 下列描述用于导航，任务目标、标注范围与组合对照以配置中的 `building_name_paper` 条目为维护源。

- [学习目标与标注范围](../../../config/task_targets.yaml)；[组合轴与对照](../../../config/combination_axes.yaml)：楼数、楼位、纸片有无、文字与视角的组合。
- [本任务数据配置](config/dataset.yaml)与[素材来源配置](config/source.yaml)：具体分支读取其支持的字段。
- [制作入口](../README.md#恢复生产入口)列出可执行分支；[任务脚本](./compose_rooftop_papers.py)是其中一个入口。产量与接线状态以[当前生产进度](../../../docs/DATASET_PRODUCTION.md#当前生产进度)及实际回执为准。
- [训练入口](../../../lab/features/fire/training/scripts/train_building_name_paper_v4.py)及[训练流程](../../../docs/TRAINING_WORKFLOW.md)：按任务配置和实际预算使用；脚本存在不代表模型已训练或完成交付。

数据制作使用同一 [dataset-production-loop](../../../.agents/skills/dataset-production-loop/SKILL.md)，无需按任务复制 Skill。

## 当前全量检测数据

本轮预算为 train 3000、val 300、test 300，分配维护于 [full_recipe.json](config/full_recipe.json)。13 张审核底图来自 4 个父视频，按父来源分集；近景车辆、桶体导致尺度不合适的候选已从整楼合成中移除。三个 split 分别覆盖 A/B/C/AB/AC/BC/ABC，以及每栋楼的有纸／无纸开关，共 26 种非空楼体状态。没有纸条的图像保留空标签，多楼部分有纸逐张标注实际可见纸面；本批只有 paper 检测监督。

`prepare_full_dataset.py` 从当前逐图资格编译三集计划，`render_full_layers.py` 输出相同机位下有纸／无纸楼体、地面光照及纸条掩膜，`build_full_dataset.py` 复用这些物理层制作全量图签。低机位为 0.14–0.23 m，左前单楼 70°、多楼 50°；底图透视按地平线及近似焦距反投影，接地阴影来自实际 Cycles 接收面。成图施加有界整帧取景、曝光和轻微模糊变化，框与最后可见掩膜同步。

```powershell
# 更新资格时才加 --register；普通续作直接编译。
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/building_name_paper/prepare_full_dataset.py --accepted-templates datasets/annotated/fire/building_name_paper_full/accepted_templates.json
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/building_name_paper/build_full_dataset.py --replace
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/building_name_paper/build_full_dataset.py --audit-only
# 修订来源索引身份、最终纸框和生成控制诊断元数据，保留全部图片和标签。
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/building_name_paper/build_full_dataset.py --refresh-source-index
```

完整物理层、初始生成输入和最后可见掩膜保存在 `datasets/annotated/fire/building_name_paper_full/`。主层与补充层由 `full_recipe.json#layer_manifests` 统一读取，当前选用 182 个物理模板；这些模板经组合与轻微成像变化扩展到预算，图片数量不代表独立拍摄场景数量。实际三集数量、来源分支、空负例、小纸条尺寸和组合核对见 [audit.json](../../exports/building_name_paper/audit.json)，训练入口为 [data.yaml](../../exports/building_name_paper/data.yaml)。val/test 保留既往开发接触，且共享楼体资产，用于诊断当前识别能力；独立真实场景效果仍需另行验证。本轮不训练模型、不创建正式发布版本。

当前训练交接使用 [楼名训练包](../../../training_packages/building_name_paper/README.md)，配置与评估顺序见[训练流程](../../../docs/TRAINING_WORKFLOW.md#楼名纸条训练包)。来源索引中的 `paper` 元数据绑定最终图签及已知生成控制，`source_reuse` 显式声明纯渲染与楼体实拍合成共享 Buildings 资产；诊断字段不补造实拍机位或 OCR 真值。

## 新视频独立微调

用户指定将独立微调数据放在 `training_packages/building_name_paper/finetune_dataset`。入口 [build_video_finetune.py](build_video_finetune.py) 只读包内三段 `buildingname*.mp4`，使用[纸面四角及抽帧配方](config/video_finetune_review.json)，以纸面特征光流和单应性传播标注，完整框核对后生成[执行计划](config/video_finetune_plan.json)。原模型预测不作为标注，原训练图不回放。窗格、火焰图案和无纸条区域生成完整空负；新视频裁剪及多视图拼接仅用于 Train。按视频文件隔离 Train/Val，但三段来自同一场景和拍摄 session，属于同场景诊断。

```powershell
# 首次编译与生成；已有输出保留，不自动覆盖
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/building_name_paper/build_video_finetune.py
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/building_name_paper/build_video_finetune.py --generate
```

实际数量和纸条框以[独立清单](../../../training_packages/building_name_paper/finetune_dataset/manifest.json)为准；训练及结果见[新视频微调](../../../docs/TRAINING_WORKFLOW.md#楼名新视频独立微调)。包内旧 `dataset`、回传权重与测试材料保留。

## 当前四分支小样

`build_current_sample.py` 将完整标注实拍、实拍底图替字、纯 Blender 与 Blender 楼体＋实拍底图图签汇总到同一个 `datasets/samples/building_name_paper/`。每张图在根清单中保留来源分支、原 split、父来源、逐图资格修订和检测框，模型输入在 `images/` 与 `labels/`，带框总览在 `contact_sheet.jpg`。本入口复用 `package_direct_real.prepare`、`compose_rooftop_papers.compose_one/augment_scene` 和楼名 Blender 渲染器。

当前为实拍 9 张、实拍底图替字 12 张、纯 Blender 31 张、Blender 楼体＋实拍底图 28 张，共 80 张、114 个纸条框和 6 张空负。实拍来自现有 5 张 train 帧及 4 张完整标注档案照片，档案照片保留 unassigned 资格及已有 val/test 接触。合成以两张 train 实拍底图的完整纸框作替字，每张底图生成 6 张，交替使用原构图与轻微整帧透视、明暗变化。原有目标框随变化更新。此小样用于开发和目视检查，集合分配与接触按清单记录。

素材选择及每底图预算在 [source.yaml](config/source.yaml) 的 `sample_recipe` 维护。已认可的 31 张低机位、左前大偏航 Blender 图像、标签、掩膜和逐状态计划保存在 `datasets/annotated/fire/building_name_paper_sample/`。根小样保留可直接读取的图签副本，复验字段回指这份持久证据。

```powershell
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/building_name_paper/build_current_sample.py --replace
```

重复运行会复用同一份已核对的 Blender 图签，并按当前实拍资格与合成配额制作小样。`--replace` 只替换旧清单拥有的生成文件，保留其余文件。已有根目录被 Windows 占用时，本入口通过可恢复的文件替换保留根目录。

## Blender 楼体与实拍底图合成

当前小样由 `source.yaml#sample_manifest` 指向实际清单；新增路线的可写配方为 [hybrid_plan.json](config/hybrid_plan.json)，保持原直接实拍和替字配方。`hybrid_sample.declaration` 将底图、七种楼数组合与两种视角展开为执行状态，生成时完整状态保存在层清单及根小样 `production_plan`。该小样的实际数量从当前清单读取；旧全局计划统计不代表本轮小样产量。

使用已接触的 P01、P12 两张完整车载实拍，逐图库中只补入本任务的完整空负标签和地面合成资格；全量编批时其父视频分配到 test，既往现场评估接触继续保留。底图中的人物、灯架脚和垃圾桶由脚点排除区保护。A/B/C/AB/AC/BC/ABC 各做低机位左前和正前对照：机高 0.14／0.20 m，单楼左前 70°、多楼左前 50°。道路两侧标线交点提供地平线；焦距、机高及无滚转假设是合成近似，不是实测标定。相机射线反投影到同一 Z=0 地面，逐楼投影整个底座并检查支撑区和画面边界。

[render_hybrid_sample.py](render_hybrid_sample.py) 读取统一工程的 Buildings，在独立后台进程中输出透明楼体、带遮挡／无楼体的漫反射地面和逐纸可见／孤立掩膜。共用的 `common/blender_photo.py` 从既有整桶实拍渲染器抽取相机射线、投影和接收面能力。World 环境光、漫反射反弹及大面积柔光产生实际 Cycles 阴影。合成在当前统一入口内执行，在线性光下将两层地面比值作用于底图，限制到真实地面；用白色标线作有界曝光和色彩近似匹配，并对预乘前景及 alpha 同步轻微柔化。框从最后可见纸面掩膜提取，合成前核对其像素数量、框及透明前景覆盖。

持久前景、阴影和掩膜保存在 `datasets/annotated/fire/building_name_paper_sample/hybrid/layers/`，成图、标签和阴影衰减在同级 `hybrid/composites/`。根 sample 保留直接可用的图签副本及证据引用，`hybrid_contact_sheet.jpg` 专看单／两／三楼，`contact_sheet.jpg` 包含完整四分支。

```powershell
# 只在新增或修订逐图地面审核时写入资格；其他任务的证据保持原样。
.venv/Scripts/python.exe -X utf8 -B -m datasets.scripts.building_name_paper.hybrid_sample --review-backgrounds
# 先 dry-run 核对全部相机、脚点和边界。
& $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/building_name_paper/render_hybrid_sample.py -- --dry-run --device OPTIX
& $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/building_name_paper/render_hybrid_sample.py -- --device OPTIX
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/building_name_paper/build_current_sample.py --replace
```

渲染层重新生成前按其清单清理本分支生成结果，保留配方、原始底图与生产回执。已有层可直接重跑统一小样入口；该入口会复验当前底图资格并重建合成和总览。

## Blender 光照、轨迹和楼体开关

统一工程的 `Buildings` 场景包含四台相机，`Camera - A`、`Camera - B`、`Camera - C` 各围绕一栋楼，`Camera - ABC` 围绕三栋楼。相机走前方球面带，单楼覆盖左前 70° 至右前 45°，三楼覆盖左前 50° 至右前 45°。三楼范围兼顾低机位下相邻楼体的遮挡。相机高度按两张车载参考的低视角和平视构图近似设计，当前为 0.14、0.20、0.23 m，楼高约 0.40 m。每条轨迹共 275 帧，1–81、98–178、195–275 帧分别为三档距离和高度，段间连续过渡。[实际参数](config/source.yaml) · [工程配置入口](configure_blender_scene.py)

纸面与油墨使用粗糙漫反射材质，照明采用 World 环境光和 Cycles 漫反射反弹。后台入口保留实际相机矩阵、机高与纸条可见掩膜，旧计划中的反光字段不会增加条灯或清漆。

机位构图参考车载帧 [f000802.jpg](../../raw/fire/targets/camera/car_real_capture_20261112/frames/f000802.jpg) 和 [f000893.jpg](../../raw/fire/targets/camera/car_real_capture_20261112/frames/f000893.jpg)。高度和镜头参数为近似设计，实际相机标定未知。

在 Blender 的大纲视图中，A、B、C 三栋楼分别在各自 Collection 内。集合的视口开关和相机图标分别控制显示和渲染。纸面与文字在所属楼的 Collection 中，可以成对开关。选择 `Buildings - ABC` 视图层并使用 `Camera - ABC`，即可通过三个楼体 Collection 组合 A、B、C、AB、AC、BC、ABC。单楼专用视图层 `Buildings - A/B/C` 隔离其余楼体。

脚本操作使用 `configure_blender_scene.select_camera(scene, 'ABC', frame)` 和 `set_buildings(scene, 'AC', papers='AC')`；将 `papers` 设为空串可保留楼体并关闭楼名纸条。渲染计划的 `background.buildings` 控制整栋楼，`target.texts` 仅列出带纸条的楼。需要重渲染 Blender 分支时，清理该分支清单拥有的生成结果并保留 `blender_plan.json`，再运行下列命令及上方统一小样入口。

```powershell
& $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/building_name_paper/render_blender_names.py -- --plan datasets/annotated/fire/building_name_paper_sample/blender_plan.json --output datasets/annotated/fire/building_name_paper_sample/blender --device OPTIX
```

火焰镜头位于同工程的 `Fire Buildings` 场景，使用独立时间轴、灯光与 Collection 开关，楼宇物体和网格仍共享。当前产量与核对结果见[生产进度](../../../docs/DATASET_PRODUCTION.md#当前生产进度)。

<!-- TASK_DESIGN:START -->

## 学习目标与组合设计

本节由共享配置生成；修改目标与组合请编辑 [task_targets.yaml](../../../config/task_targets.yaml) 和 [combination_axes.yaml](../../../config/combination_axes.yaml)，运行 `python lab/scripts/contracts/state_coverage.py --write`。

**学习及标注**

- 类别：paper
- 文字：text
- 数量：zero、single、multi
- 负例：zero

纸片能定位就保留检测标签；文字不可读时 OCR 记未知；排版和透视合成保留正常文字方向。

纸片有无、纸片实例数和已标注的文字条目数独立统计；同名多纸逐纸计数。

**素材用途**

- `direct_target`：至少一张纸片且检测框完整；OCR 层另取图中可读且全文已知的逐纸记录。
- `target_texture`：已知全文的纸面图案、合规字体或程序生成文本，保留与可定位纸片的对应。

**组合轴与对照**

以下是本任务的设计条件；`dynamic` 表示依据实际素材和任务计划确定。

- `building_count`：single、multi
- `yaw`：frontal、oblique

**布局**：楼面 ID、楼顶贴放区域及坐标尺寸；用槽位关系区分新增纸、替换旧纸和保留其他纸。

**building_text**：楼数、楼位和纸片惯用位置不能决定有无纸或文字。

- 安排：单楼和多楼内安排有／无纸；同一个楼位轮换文字，同一文字跨楼位与可辨视角。
- 核对：按样本清单 buildings 和最终纸框、四点、文字记录核对；逐纸计实例，逐条计有文本的 OCR 标注，同名两纸仍是两个实例。

对照描述来自设计配置；结构化记录目前为部分编码，具体制作按任务入口和可用素材安排。

<!-- TASK_DESIGN:END -->
