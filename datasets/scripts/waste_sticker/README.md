# 垃圾物品贴纸数据任务

本任务为 `waste_sticker`。定位并识别固定 24 类物品图案；需要时另记桶归属。 下列描述用于导航，任务目标、标注范围与组合对照以配置中的 `waste_sticker` 条目为维护源。

- [学习目标与标注范围](../../../config/task_targets.yaml)；[组合轴与对照](../../../config/combination_axes.yaml)：载体、位置、尺度、视角、光照和遮挡。
- [本任务数据配置](config/dataset.yaml)与[素材来源配置](config/source.yaml)：具体分支读取其支持的字段。
- [制作入口](../README.md#恢复生产入口)列出可执行分支；[任务脚本](./generate_sticker_real_photo_v4.py)是其中一个入口。产量与接线状态以[当前生产进度](../../../docs/DATASET_PRODUCTION.md#当前生产进度)及实际回执为准。
- [训练入口](../../../lab/features/waste_bin/training/scripts/train_waste_sticker_v4.py)及[训练流程](../../../docs/TRAINING_WORKFLOW.md)：按任务配置和实际预算使用；脚本存在不代表模型已训练或完成交付。

当前小样从 `source.yaml#sample_manifest` 定位，图签、实际数量和组合由该清单维护。旧独立渲染和指定实拍配方通过 `legacy_blender_plan`、`legacy_direct_real_plan` 保留历史参考用途，退出当前覆盖计划列表；正式分集读取 `full_train_plan.json`、`full_val_plan.json` 和 `full_test_plan.json`。

数据制作使用同一 [dataset-production-loop](../../../.agents/skills/dataset-production-loop/SKILL.md)，无需按任务复制 Skill。

`source.yaml` 的 `real_source_catalog` 中，`annotation` 和 `gap` 记录旧三行 `real_annotations` 清单的状态；逐帧机器视觉结论查 `frame_environment`。当前直接实拍计划与可用未分配帧见[实拍补录](../../../docs/DATASET_PRODUCTION.md#当前生产进度)。

## 白色背景硬负例微调

[当前计划](config/recall_preserving_finetune_plan.json)与现行 `source.yaml` 的 `finetune_plan`／`finetune_manifest` 接线；该专项与完整基础集合分别保留。完整回放原 Train，加入经全帧机器视觉审查的 Reshoot 无贴原图及其裁剪、全 24 类按输入像素分层的白卡／真图案对照，以及逐图库 Train 硬负。模型预测只用于排序，不作为空负真值；Reshoot 整场只进入 Train。已有输出存在时构建器拒绝覆盖；`--plan` 可显式指定配方，默认读取当前 `source.yaml#finetune_plan`。

```powershell
.venv/Scripts/python.exe -X utf8 -B -m datasets.scripts.waste_sticker.build_hard_negative_finetune mine
.venv/Scripts/python.exe -X utf8 -B -m datasets.scripts.waste_sticker.build_hard_negative_finetune build
```

当前产量和逐图证据读取 [专项清单](../../annotated/waste_bin/waste_sticker_recall_preserving/manifest.json)，训练参数及运行位置见[微调交接](../../../docs/TRAINING_WORKFLOW.md#垃圾贴纸白色背景硬负例微调)。上一轮配方和清单由 `legacy_finetune_plan`／`legacy_finetune_manifest` 保留；已执行运行继续使用当时快照。

<!-- TASK_DESIGN:START -->

## 学习目标与组合设计

本节由共享配置生成；修改目标与组合请编辑 [task_targets.yaml](../../../config/task_targets.yaml) 和 [combination_axes.yaml](../../../config/combination_axes.yaml)，运行 `python lab/scripts/contracts/state_coverage.py --write`。

**学习及标注**

- 类别：0、1、2、3、4、5、6、7、8、9、10、11、12、13、14、15、16、17、18、19、20、21、22、23
- 数量：empty、single、multi
- 有无：present、absent
- 负例：empty

按可辨图案标检测框和类别。联合投放评估时另记可确认的桶归属；类别或归属不可判时各记未知。

贴桶正例每图 1–4 个可见桶，按标签完整的贴纸分 0 至 4 个计图，每桶至多一贴，按 24 类逐贴计有效实例。硬负例可含 0–4 桶；空白卡不新增类别，真白底贴纸仍完整标注。桶色作为场景变化和下游桶类属性。现场下游按位置关联桶与贴纸，再按桶类与物品集合判定投放。

**素材用途**

- `direct_target`：至少一张任务范围内的指定物品贴，且全帧范围内各目标的检测框和类别完整。可使用桶面、其他表面或独立摆放的贴纸实拍。
- `target_texture`：24 张固定物品图案及类别索引，可用于合格的实拍表面或 Blender 载体。
- `hard_negative`：整帧有证据确认无任务范围内物品贴，覆盖有桶无贴、固有分类图标、白色道路标线、空白纸卡和相似条纹。模型预测只用于排序，不作为空负真值。

**组合轴与对照**

以下是本任务的设计条件；`dynamic` 表示依据实际素材和任务计划确定。

- `carrier`：bin_face、other_surface、standalone
- `background`：dynamic
- `position`：dynamic
- `scale`：dynamic
- `rotation`：dynamic
- `view`：frontal、oblique
- `lighting`：normal、dim
- `occlusion`：none、partial

**布局**：贴纸可独立摆放，也可贴在合适的表面。贴桶的合成分支使用已确认的桶面可贴区及固有标志避让区，保留其他可辨物品贴。

**carrier_position**：贴纸在不同载体和摆位下仍能凭图案识别。

- 安排：同一图案轮换桶面、背景、面内位置与旋转；同一桶面换图案。覆盖 1 至 4 个可见桶、不同贴纸大小、正斜视角、光照和部分遮挡，桶色按场景自然变化。
- 核对：逐贴核对 class_id、可见框和父来源，按可见贴纸数及负例分布核对图数，按 24 类分别核对有效实例数。桶体颜色保留下游归属和投放用途。

**white_background_counterexamples**：区分真实物品图案与白色道路标线、空白纸卡及相似条纹，保留白底真贴纸的召回。

- 安排：相同背景制作无贴与有贴对照，相同桶面制作空白卡与指定图案对照；成对派生留在同一 split，实拍父视频按集合隔离。正例保留全部可见贴纸，空负继承完整无目标证据。
- 核对：固定置信度下分别统计白线与空白卡每百图误检框数、带误检图比例及 24 类召回；旧 Test 仅作已知回归，新的挑战图不进入 Train。

对照描述来自设计配置；结构化记录目前为部分编码，具体制作按任务入口和可用素材安排。

<!-- TASK_DESIGN:END -->
