# 视觉识别资料与版本对照（2026-11-20）

本页把国赛研发仓库、本公开仓库与相邻的 `AIAEC-VisualRecognition` 工作空间的入口连在一起。数字按原记录的观察时间和样本范围陈述；相邻工程独立产生的训练和数据不计作国赛 V4 的新版本。模型权重、原始图像和完整运行目录不随本仓库发布。

## 资料与实现入口

| 内容 | 本仓库入口 | 相邻工作空间入口及用途 |
| --- | --- | --- |
| 识别任务与系统设计 | [视觉架构](VISION_ARCHITECTURE.md)、[任务矩阵](TASK_MATRIX.md)、[异步识别](ASYNC_PERCEPTION_DESIGN.md) | [`docs/BLENDER_SITE_RECONSTRUCTION.md`](https://github.com/101AoiKoori/AIAEC-VisualRecognition/blob/main/docs/BLENDER_SITE_RECONSTRUCTION.md)记录场地建模研究；各任务 `task.yaml` 定义该空间任务 |
| 数据制作 | [生产流程](DATASET_PRODUCTION.md)、[设计](DATASET_DESIGN.md)、[脚本入口](../datasets/scripts/README.md) | 各任务 `sources/`、`materials/`、`annotations/`、`datasets/` 的清单记录来源与分集；两空间的图数不可直接相加 |
| 推理与训练 | [`ws/src/aec_vision/`](../ws/src/aec_vision/README.md) 是共享推理核心；[`lab/features/`](../lab/features/README.md) 放实验入口；[训练流程](TRAINING_WORKFLOW.md) | 相邻空间的 `*/scripts/infer/`、`*/scripts/train/` 与 `*/experiments/` 是该空间独立研究入口，未直接复制到 ROS 部署包 |
| 模型与历史 | [模型目录说明](../models/README.md)、[阶段报告](research/DATASET_MODEL_REPORT_20261119.md)、[`Archive/technical/`](../Archive/technical/README.md) | [`docs/MODEL_SYNC.md`](https://github.com/101AoiKoori/AIAEC-VisualRecognition/blob/main/docs/MODEL_SYNC.md)登记八份选定权重的迁机核对；各任务 `models/*/model.json` 及运行记录解释父模型与版本 |

公开仓库中的代码、配置和文档提供方法与复现入口。推理分数依赖具体权重、输入、阈值和场景；未提供权重的公开克隆不能直接复现下表。

## 国赛 V4 数据规模与效果

截至 11 月 19 日的实际导出，人物为 3000/300/300、指针仪表 6480/1080/1080、火焰贴纸 3000/300/300、楼名纸条 3000/300/300、禁行牌 2500/300/300、物品贴纸 1800/240/240（顺序均为 Train/Val/Test）。桶开闭完整基线为 3500/186/253，后续真实桶主导微调输入为 804/186/253；电动车冻结首轮基线为 108/10/未建 Test。这些是图片数，实例数、空标签数及来源范围见[阶段报告](research/DATASET_MODEL_REPORT_20261119.md#数据集现状)。

同一报告保存六项完整训练的 Val/Test mAP50–95、TP/FP/FN 和模型摘要。下表只列可配对预测的微调前后对照；正确框置信度是检出框的中位数，未检出的真值没有置信度。使用 conf≥0.25、匹配 IoU≥0.5 的原记录口径，不能把置信度当准确率。

| 对照范围 | 方法 | 正确框置信度中位数 | TP / FP 前→后 | 结论 |
| --- | --- | --- | --- | --- |
| 人物，53 张已知开发图 | 窄亮边 10 轮微调 | 0.9131→0.9204（+0.0073） | 83/29→83/4 | FP 减 25；开发集有来源复用 |
| 火焰贴纸，300 张 Test | 精简 Train、10 轮微调 | 0.9475→0.9389（−0.0087） | 475/9→480/4 | 召回增加 5 个，分数中位数略降 |
| 楼名纸条，21 张新同场景 Val | 视频专项 43 轮 | 0.4179→0.8570（+0.4391） | 4/2→10/0 | 旧 Test 的整图全对反而 268/300→264/300 |
| 垃圾桶开闭，253 张 Test | 真实桶主导 10 轮 | 0.9834→0.9706（−0.0128） | 388/2→387/43 | 整图全对 225/253→193/253，候选未替代基线 |
| 物品贴纸，360 张挑战 Test | 保召回微调 10 轮 | 0.9542→0.9638（+0.0096） | 575/162→573/31 | FP 减 131，同时漏检增 2 |

楼名旧 Test 的正确框中位数另为 0.9364→0.8836，TP/FP 为 333/50→334/38。完整分母、同一真值的配对差、P10/P90、训练逐轮波动和原回执摘要见[置信度证据](research/DATASET_MODEL_REPORT_20261119_CONFIDENCE.json)。指针仪表与禁行牌没有同输入的前后配对预测，不给出置信度变化值。此处的“之前／之后”指对应历史运行及专项微调，不等于相邻工作空间 11 月 20 日的当前现场模型。

## 相邻工作空间 11 月 20 日新增观察

- 电动车新增 80 对训练候选图签：66 张原图、14 张安全裁区，共 453 个直立／倒伏框。两方向各做 20 轮新增图加旧图回放；在原 36 张、252 框的已暴露同停车区开发集上，原始检测逐车严格全对 28→30/36，固定范围加复核 33→33/36；组合 FP 3→2、FN 仍为 2、匹配姿态错 0→1。改善 2 图、退化 2 图，旧权重仍是参照。来源见[电动车任务状态](https://github.com/101AoiKoori/AIAEC-VisualRecognition/blob/main/e_bike/STATE.md)及 `e_bike/experiments/v001_parking_pose_study/row_finetune_notes.md`。本轮没有同口径逐框置信度统计，故不把 28→30 写成置信度提升。
- 桶开闭另试了共享特征分离式、端到端联合式，以及原图高分辨率桶口裁块。相邻空间记录：同一 23 张留出视频，原图裁块两组均定位和状态 92/92、严格四桶 23/23；两类 YOLO 基线 22/23。另 3 张近距车载图均有顶部背景误报，未选新现场权重。见[桶开闭任务状态](https://github.com/101AoiKoori/AIAEC-VisualRecognition/blob/main/waste_bin_state/STATE.md)。23 张来自同一视频，不能解释为 23 个独立场景。
- 电表屏幕检测的独立数据入口为 3950 张 Train、1000 张 Val。相邻空间 `meter/STATE.md` 记载 r003 从初始权重以 batch 32 重新训练，模型快照的第 74 轮选择与字节校验；该任务与本仓库的八点指针仪表是不同识别目标，不并入上表的仪表结果。

## 制作与版本口径

国赛 V4 从任务学习目标、组合轴和来源配置确定监督范围；实拍、实拍底图合成和 Blender 渲染形成图签，按父来源管理 Train/Val/Test，再做基线训练。根据固定样本上的漏检、误报、姿态或状态错误制作专项补样，微调后按原样本与原阈值配对比较。相关邻帧、同一实体图案和共享母场景不能计作独立实景验证。具体步骤和当前入口见[生产流程](DATASET_PRODUCTION.md)、[训练流程](TRAINING_WORKFLOW.md)及[阶段报告](research/DATASET_MODEL_REPORT_20261119.md)。

Git 提交、模型 `v` 编号、数据清单及一次实验运行是不同的版本对象。国赛历史技术证据按 [`Archive/technical/`](../Archive/technical/README.md) 和 Git 查找；相邻工作空间的独立成果保留其自身编号与来源。本页为资料索引和已有证据的解释，没有创建正式数据版本、模型发布标签或部署版本。
