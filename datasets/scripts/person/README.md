# 人物立牌数据任务

本任务为 `person`。区分可辨人物立牌的社区／非社区类别，白背作为硬负例。 下列描述用于导航，任务目标、标注范围与组合对照以配置中的 `person` 条目为维护源。

- [学习目标与标注范围](../../../config/task_targets.yaml)；[组合轴与对照](../../../config/combination_axes.yaml)：背景、脚点、视角及人与电动车遮挡。
- [本任务数据配置](config/dataset.yaml)与[素材来源配置](config/source.yaml)：具体分支读取其支持的字段。
- [制作入口](../README.md#恢复生产入口)列出可执行分支；[任务脚本](./build_full_person_plan.py)是其中一个入口。产量与接线状态以[当前生产进度](../../../docs/DATASET_PRODUCTION.md#当前生产进度)及实际回执为准。
- [训练入口](../../../lab/features/person/training/scripts/train_person_v4.py)及[训练流程](../../../docs/TRAINING_WORKFLOW.md)：按任务配置和实际预算使用；脚本存在不代表模型已训练或完成交付。

数据制作使用同一 [dataset-production-loop](../../../.agents/skills/dataset-production-loop/SKILL.md)，无需按任务复制 Skill。

现有人物混合计划依赖可用合成布局，直接实拍可从[完整标签打包入口](../common/package_direct_real.py)单独准备。[人物生产入口](../README.md#人物最小生产路径)

<!-- TASK_DESIGN:START -->

## 学习目标与组合设计

本节由共享配置生成；修改目标与组合请编辑 [task_targets.yaml](../../../config/task_targets.yaml) 和 [combination_axes.yaml](../../../config/combination_axes.yaml)，运行 `python lab/scripts/contracts/state_coverage.py --write`。

**学习及标注**

- 类别：community、noncommunity
- 数量：empty、single、multi
- 负例：empty、white_back

只给现行 PERSON-TARGET-SCOPE 范围内、正面身份仍可辨的立牌写两类可见框。白背为硬负，不写框或忽略区域；纯侧背、薄边及身份不可判者按任务范围处理。放置尺寸与遮挡后框高分别记录。

覆盖纯空、单人、多人、仅白背及正背同框。合成编译器仍读取 2 至 6 人控制列表；直接实拍为来源，街区与跨帧去重由下游汇总。

**素材用途**

- `direct_target`：全帧可辨正面立牌的类别和可见框完整，至少一个正例。
- `target_foreground`：已确认社区／非社区身份的原图裁片或正面立牌纹理，含可见轮廓及放置依据。
- `hard_negative`：已确认白背区域；整图作纯负例需确认没有其他可辨正面。

**组合轴与对照**

以下是本任务的设计条件；`dynamic` 表示依据实际素材和任务计划确定。

- `background`：dynamic
- `position`：dynamic
- `view`：front、oblique
- `mutual_occlusion`：none、present
- `vehicle_occlusion`：none、upright、fallen

**布局**：在原图坐标核对地面支撑、目标尺寸和遮挡，保留背景中的正面框和白背区域。尺度缺少实测时记录近似依据。

**category_context**：立牌类别不能由背景、脚点或车的姿态猜出。

- 安排：同背景内安排两类可辨正面，交换脚点；让两类各出现于无遮挡和直立／倒伏车有效遮挡。
- 核对：按人物实例将类别、背景父组、脚点、可见框与遮挡者连接；用 actor ID 关联重叠，画面中有车不能单独证明挡住了谁。

**front_back**：立牌轮廓或白板不能代替正面类别特征。

- 安排：同尺寸段的可辨正面和白背单独出现，再用白背加另一块可辨正面作对照；单列真实薄边／纯侧背。
- 核对：用清单白背位置、真实视面和最终 YOLO 框核对三种场景；仅白背为完整空标签，正背同框只标可辨正面。

对照描述来自设计配置；结构化记录目前为部分编码，具体制作按任务入口和可用素材安排。

<!-- TASK_DESIGN:END -->
