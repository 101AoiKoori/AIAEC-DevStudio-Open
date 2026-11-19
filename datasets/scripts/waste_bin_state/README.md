# 垃圾桶开闭数据任务

本任务为 `waste_bin_state`。逐桶定位并区分桶盖开闭，颜色和桶身份另记。 下列描述用于导航，任务目标、标注范围与组合对照以配置中的 `waste_bin_state` 条目为维护源。

- [学习目标与标注范围](../../../config/task_targets.yaml)；[组合轴与对照](../../../config/combination_axes.yaml)：桶色、位置、视角与邻桶遮挡。
- [本任务数据配置](config/dataset.yaml)与[素材来源配置](config/source.yaml)：具体分支读取其支持的字段。
- [制作入口](../README.md#恢复生产入口)列出可执行分支；[任务脚本](./generate_waste_v4_grouped.py)是其中一个入口。产量与接线状态以[当前生产进度](../../../docs/DATASET_PRODUCTION.md#当前生产进度)及实际回执为准。
- [训练入口](../../../lab/features/waste_bin/training/scripts/train_waste_bin_state_v4.py)及[训练流程](../../../docs/TRAINING_WORKFLOW.md)：按任务配置和实际预算使用；脚本存在不代表模型已训练或完成交付。

数据制作使用同一 [dataset-production-loop](../../../.agents/skills/dataset-production-loop/SKILL.md)，无需按任务复制 Skill。

## 补拍微调数据

补拍的逐图轮廓和开闭记录由 `prepare_reshoot_sources.py` 读取已有标注配置生成，用户原件保留在 `training_packages/waste_bin_state/Reshoot`，工作副本和证据分别进入 `datasets/raw/waste_bin_state/reshoot/` 与 `datasets/annotated/waste_bin_state/reshoot/`。整桶框包含附着盖板及把手；截边图仅用于直接训练，完整轮廓才进入前景合成。

`build_reshoot_finetune.py --publish` 读取 [当前微调计划](config/reshoot_finetune_plan.json)，混合补拍、真实桶前景合成和旧 Train 回放，归档上一份导出后更新当前 `datasets/exports/waste_bin_state/`。原 Val/Test 的图像与标签保持一致；同一补拍 session 全部归 Train。首次输入记录保存在 `datasets/annotated/waste_bin_state/reshoot_finetune/baseline_manifest.json`，重跑继续使用该基线，避免把上次派生图重新混进回放。

实际配比读取当前导出清单和微调源清单，计划以 `fine_tune` 用途登记，不抵扣正式全量制作预算。微调配置为 `lab/features/waste_bin/training/config/waste_bin_state_reshoot_finetune_v4.yaml`，以用户回传的 best 初始化新优化器。制作与检查不启动训练；原 Val/Test 已用于失败分析，后续对比解释为已知来源回归。

<!-- TASK_DESIGN:START -->

## 学习目标与组合设计

本节由共享配置生成；修改目标与组合请编辑 [task_targets.yaml](../../../config/task_targets.yaml) 和 [combination_axes.yaml](../../../config/combination_axes.yaml)，运行 `python lab/scripts/contracts/state_coverage.py --write`。

**学习及标注**

- 类别：open、closed
- 数量：empty、single、multi
- 负例：empty

对任务范围内的桶逐个定位，依据可见盖板或桶口判开闭。远处非主体且已知在范围外的桶可选记 ignore。范围内的桶盖被遮挡或过渡态不可判时保留实例，状态记未知。

空／单／多桶按任务范围内可识别的桶计数。主体为桶的画面按直接目标素材使用，不进入桶开闭常规合成底图池。非主体的范围内桶带完整标签可随底图合成，域外物体按实际占位核对。空负例由画面证据确认。

**素材用途**

- `direct_target`：全帧至少一个任务范围内可识别的桶。主体桶及同画面内其他符合范围的桶均需可见框和开闭状态，另存桶色与逐桶身份。远处非主体且不属于识别范围的桶可保留为背景。
- `target_foreground`：有逐桶颜色、开闭和掩膜的实拍或渲染前景，保留桶底接地及盖口几何。

**组合轴与对照**

以下是本任务的设计条件；`dynamic` 表示依据实际素材和任务计划确定。

- `color`：blue、red、green、gray
- `position`：dynamic
- `view`：normal、low
- `neighbor_occlusion`：none、present

**布局**：桶底可用支撑区域、桶组尺寸与邻桶遮挡，保留已有桶标签。可用的尺度证据用于细化布局，缺失时记录计划尺寸或其他近似依据。

**lid_color_position**：桶色、固定桶位或邻桶状态不能替代桶盖几何。

- 安排：每种桶色各配开／闭，同桶同机位只改变盖子；多桶画面固定机位交换开闭分配。
- 核对：用实际每桶 color／ID 与开闭标签交叉并核对位置，不能用图内存在某色及某类推断两者属于同一个桶。

**view_occlusion**：低视角或邻桶遮挡不能直接提示开闭。

- 安排：同色开／闭分别配正常与低视角，无挡与局部邻挡优先成对。
- 核对：同一桶实例查盖体开闭证据、机位和实际邻挡区域；盖口不可判断时按标注规则记录。

对照描述来自设计配置；结构化记录目前为部分编码，具体制作按任务入口和可用素材安排。

<!-- TASK_DESIGN:END -->
