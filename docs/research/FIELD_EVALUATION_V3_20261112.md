# 2026-11-12 同场地实拍诊断与 V4 训练设计

> 本文提及的部分本地素材、数据和历史运行产物未随本仓库提供；相关名称保留用于说明来源和实验背景。

本报告记录旧 `site_capture_20261112` 批次的同场地评估。新 `CarRealPicture` 批次已纳入 V4 核心来源，尚未计入下述预测或指标。车载照片是原生 640×480，两段补录视频保留各自原始视角、分辨率和时间戳。候选标注在看图阶段独立制作。本阶段使用已入库素材，按可确认的原始来源和布置保留验证、测试及实拍诊断样本。有关原始媒体和抽帧，见[实拍批次说明](../../datasets/SITE_CAPTURE_20261112.md)。

## 筛选与评分

160 张车载图筛出 31 张，122 张留待复核，5 张标记重复或近重复，2 张因视野被挡或镜头偏离观察区排除。重复候选用同构图缩略图相似度筛查，再目视核对，其中两对解码像素完全相同。388 张抽帧保留 29 张代表图，359 张有区段级任务线索待复核。视频帧两两查重后没有达到合并条件的图，入选的 899 对照片与视频帧也未达到跨来源缩略图近重复阈值。照片和视频里的相同物料仍需按物理实例或采集批次归组。

60 张代表图保留原始清晰度，共有 107 条图像与任务记录及 132 个候选框。完整候选正例 47 条，目视负例 35 条，部分标注 25 条。候选框在原图坐标中逐图估计。只使用画面内该任务目标可确认完整的候选记录评分，部分标注留待复核。未选入的画面没有参与计分。详情见[评估集清单](../../datasets/annotated/site_capture_20261112/field_eval_v1/field_eval_manifest.json)和[合并候选标注](../../datasets/annotated/site_capture_20261112/field_eval_v1/evaluation_candidates.json)。

本次纳入的七个生产 V3 `best.pt` 各在 548 张图上推理。使用 Ultralytics 8.3.203、PyTorch 2.7.1+cu128、RTX 4060 Laptop GPU、输入尺寸 640 和 NMS IoU 0.70。保存置信度至少 0.01 的预测，按固定置信度 0.25 和同类别一对一匹配 IoU 0.50 评分。56 条具备完整候选标注和权重的图像与任务记录计分。诊断样本按可见目标和失效条件选取，数字是该样本的描述统计，人工复核会改变候选框及分母。逐图记录另列车载照片、全场景视频和人物视频。[推理环境与检查点](../../runs/site_capture_20261112_v3_eval/prediction_summary.json) · [评分汇总](../../runs/site_capture_20261112_v3_eval/score_conf025/summary.json) · [逐图诊断](../../runs/site_capture_20261112_v3_eval/score_conf025/per_image.json)

## 实拍失效

| 模型 | 计分图／负图 | TP／FP／FN | 可复核的观察 |
| --- | ---: | ---: | --- |
| 楼名纸 | 9／4 | 2／7／3 | 正面有检出。斜视楼面框偏离，移动软化样本同类重叠框置信度仅 0.043。4 张负图中 2 张出现误报。[斜视样例](../../runs/site_capture_20261112_v3_eval/score_conf025/previews/name_slope.jpg) |
| 电动车 | 8／2 | 3／3／7 | 低视点三车全漏，近景的截断倒伏车和同框直立车均未检出。人物与车叠加时还有额外框。[低视点样例](../../runs/site_capture_20261112_v3_eval/score_conf025/previews/bike_low_view.jpg) · [近景样例](../../runs/site_capture_20261112_v3_eval/score_conf025/previews/bike_truncated.jpg) |
| 火焰贴纸 | 10／5 | 16／10／1 | 斜视小贴纸漏检，单张贴纸附近有重复框。红灯和黄灯各被报为贴纸，置信度 0.83 和 0.65。5 张负图中 2 张误报。[灯具误报](../../runs/site_capture_20261112_v3_eval/score_conf025/previews/fire_negative.jpg) · [重复框](../../runs/site_capture_20261112_v3_eval/score_conf025/previews/fire_duplicates.jpg) |
| 禁行牌具 | 11／3 | 6／6／2 | 车载候选正例均有匹配框，视频里的白墙低反差背面漏检。近景截断时模型框与整牌具口径的 IoU 仅 0.21。方向和路线分支另需核验。[截断样例](../../runs/site_capture_20261112_v3_eval/score_conf025/previews/sign_truncated.jpg) |
| 桶开闭 | 6／1 | 6／9／14 | 14 个漏配中有 7 个与另一状态类别框高重叠。示例中的闭合桶被高置信报为开启，须核对盖板标签并补不同颜色与开闭组合。[状态错报](../../runs/site_capture_20261112_v3_eval/score_conf025/previews/bin_wrong_state.jpg) |
| 垃圾贴纸 | 5／5 | 0／0／0 | 旧冻结诊断计分的 5 张均为带桶类别标志的干扰负例，均未误报；该诊断没有物品贴纸正例召回结果。新车载贴图桶另建审核与评分清单。 |
| 仪表 | 7／3 | 0／0／4 | 反光、斜视和近正视的双针表框均未检出。4 个候选正例和全部 548 张输入在置信度 0.01 以上都没有仪表预测。本轮只评表框。[车载样例](../../runs/site_capture_20261112_v3_eval/score_conf025/previews/meter_vehicle.jpg) · [近正视样例](../../runs/site_capture_20261112_v3_eval/score_conf025/previews/meter_front.jpg) |
| 人物 | 未评分 | 缺生产 V3 | 当时记录 5 条完整负例和 14 条部分标注。复用前按当前近距正面范围重审，白背作为硬负例；本表旧标签及未评分状态保持原实验口径。 |
| 车牌检测 | 未评分 | 本次未纳入 | 14 条完整正例、7 条负例和 2 条部分标注。车牌 V3 权重已回传，国赛车载评估待执行。OCR 与车位归属仍待核。 |

仪表在同一个推理环境里对 4 张原合成验证图均给出表框，置信度约 0.947–0.960。该对照证实检查点在原数据域可以输出，实拍测试覆盖的画面表现与合成验证差距明显。应先增加真实仪表尺度、玻璃反光、背景和表盘外观的训练覆盖，随后复核八点与读数，单调置信度无法补回这一批缺失的框。[合成对照](../../runs/site_capture_20261112_v3_eval/meter_synthetic_control.json)


## 五项模型的分辨率与目标尺度

按固定种子 42 从五项实际 V3 运行的训练数据各抽 512 张，读取图像头和原标签。用长边 640 等比例折算目标框像素，折算值对应未施加 Mosaic 等增强的整图。楼名、火焰的抽查原图为 640×640，电动车、禁行牌具和桶开闭为 960×480。五项模型的训练及实拍推理尺寸均设为 640。推理时 640×480 车载内容等比例进入方形输入，空余区域补边。判断小目标覆盖需比较框内实际像素与训练分布。

| 模型 | 已观察的训练与实拍目标像素差别 | 优先检查 |
| --- | --- | --- |
| 楼名纸 | 抽样训练框中位约 75×20 px，计分车载正例约 51–114×32–38 px。两张相近视频帧的框约 131×41 和 128×40 px，前者检出，后者重叠候选置信度 0.043。 | 斜视、运动软化、纸条定位以及可读性。 |
| 电动车 | 训练直立、倒伏类中位约 45×55、70×36 px。低视点三车漏检，折算最小框仍约 87×148 px。 | 低机位、倒伏、截断和人车重叠。 |
| 火焰贴纸 | 训练框中位约 16×23 px。实拍唯一漏检约 21×30 px，同图约 14×29、16×25 px 的贴纸被检出。 | 灯具误报、重复框及远小贴纸。 |
| 禁行牌具 | 车载 6 个正例均有匹配框，视频中的低反差背面漏检，截断近景约 336×360 px 的整牌具框匹配 IoU 0.21。 | 背面反差、裁切及整牌具框口径。 |
| 桶开闭 | 训练开、闭类框中位约 98×135–139 px，车载候选约 83–128×126–171 px，像素尺度重合。 | 盖板状态标注、观察角度及桶色与开闭组合。 |

五项的实拍现象覆盖多个因素。相近尺度下既有检出也有漏检，且误报和错类出现在尺度重合的样本。分辨率的单独影响可在同一批冻结图上保持权重、匹配及其余推理设置一致做输入尺寸对照。原生图的细节上限由拍摄决定。以上范围来自少量已标候选，逐图尺寸与预测可回查[逐图诊断](../../runs/site_capture_20261112_v3_eval/score_conf025/per_image.json)。

## V3 训练行为

| 模型 | epoch 上限 | 日志最佳轮 | 早停轮 | 末轮学习率 | 训练曲线 |
| --- | ---: | ---: | ---: | ---: | --- |
| 楼名纸 | 150 | 9 | 39 | 0.000851 | [CSV](../../runs/building_name_paper/runs/building_name_paper/v3_20261111_143524_607561/training/results.csv) |
| 电动车 | 150 | 60 | 90 | 0.000362 | [CSV](../../runs/e_bike/runs/e_bike/v3_20261111_142947_802052/training/results.csv) |
| 火焰贴纸 | 150 | 27 | 57 | 0.000697 | [CSV](../../runs/fire_sticker/runs/fire_sticker/v3_20261111_144145_539022/training/results.csv) |
| 禁行牌具 | 120 | 21 | 51 | 0.000633 | [CSV](../../runs/sign/runs/sign/v3_20261111_150300_452905/training/results.csv) |
| 桶开闭 | 120 | 13 | 43 | 0.000730 | [CSV](../../runs/waste_bin_state/runs/waste_bin_state/v3_20261111_151012_816475/training/results.csv) |
| 垃圾贴纸 | 160 | 56 | 86 | 0.000456 | [CSV](../../runs/waste_sticker/runs/waste_sticker/v3_20261111_151346_593593/training/results.csv) |
| 仪表 | 180 | 39 | 69 | 0.000690 | [CSV](../../runs/meter/runs/meter/v3_20261111_145506_148167/training/results.csv) |
| 车牌检测 | 120 | 76 | 106 | 0.0000477 | [CSV](../../runs/plate_detection/plate_detection/v3_20261111_152409_363091/training/results.csv) |

八项均在验证 fitness 的最佳轮之后连续 30 轮未刷新而正常早停。表中日志最佳轮是早停耐心基准。配置使用 AdamW、初始学习率 0.001、3 轮 warmup、余弦下降和末端系数 0.01。Ultralytics 8.3.203 将动量参数 0.937 用作 AdamW 的 β1。表中末轮为日志 `lr/pg0`，均未到计划终点 1e-5。七项检测的 Mosaic 计划在训练最后 10 轮关闭，早停发生在此前。仪表从开始就关闭 Mosaic。[Ultralytics 8.3.203 训练源码](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/engine/trainer.py)

回传的车牌 V3 有完整训练状态、`best.pt`、`last.pt` 与省赛合成验证。592 张验证图、1,367 个目标的 Box mAP50–95 为 0.995。该回传运行只完成验证，国赛车载回归及已暂存的 CCPD 测试没有对应结果。前述七项冻结实拍统计保持其原评分范围。[车牌训练状态](../../runs/plate_detection/plate_detection/v3_20261111_152409_363091/metadata/train_status.json) · [车牌验证](../../runs/plate_detection/plate_detection/v3_20261111_152409_363091/evaluation/val/summary.json)

框检测按 Box mAP50–95 选最佳权重，仪表按 Box 与 Pose mAP50–95 的和选最佳权重。工作空间的外层调参默认按 Pose mAP50–95 比较仪表试验。后续把训练中保存候选与外层选模的比较指标统一，并另报任务实际读数。楼名 Box mAP50–95 从最佳轮 0.633 降至末轮 0.360，说明继续同样配置训练未保持最佳验证表现。火焰与楼名在合成验证集的 0.25 阈值下已有明显误报，需用实拍错误回看相似背景、贴纸尺寸和标签口径。[Ultralytics 8.3.203 指标源码](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/utils/metrics.py)。

## V4 数据入口

当前首要实拍来源为 [CarRealPicture 新车载批次](../../datasets/CAR_REAL_CAPTURE_20261112.md)，包含视频、逐帧序列、抽样子集和任务点照片，并有贴上物品图片的垃圾桶。新批次已组织片段和原清晰度诊断候选，本轮重新推理与候选评分单列在该批次记录；与本文旧分母分开。完成类别、框、时序和归属复核后建立 V4 的逐任务开发与测试集合。原清晰度样本用于新基线和入选模型回归，增强与合成另记来源。旧评估集保留其已测范围，新素材的覆盖与效果独立报告。

各任务选材、标注和验收见[V4 数据制作方案](../DATASET_DESIGN.md#v4-八项任务数据集设计)，已生成的导出和候选配置见[当前素材与来源候选](../../datasets/README.md#当前素材与来源候选)。已冻结的 60 张诊断图用于回归，原文件名推定的 11 月 11 日 12:05:51–12:09:18 车载路线独立登记为适配候选。两组仍有同场地、同物料关系，实际会话身份未核。

人物与车牌的既有诊断记录集中在[优先任务复核队列](../../datasets/annotated/site_capture_20261112/field_eval_v1/priority_review.json)。车牌视频近景多次拍到的 3 个物理牌照，候选转写为 `京G·V7T6J`、`吉T·WC818`、`豫E·UW2Q3`，依据现有清晰车载近景和视频裁剪核对。历史同车型立牌使用了不同字符串。现场 OCR 评估保留可读候选转写和物理牌分组，旧候选清单未填写车位归属真值；后续按新任务点照片和对应观察段核对归属。

## 有界训练对照

逐模型的验证回落、分布偏移和样本规模判断见[覆盖与拟合研究](DATASET_COVERAGE_TRAINING.md#2-目前的拟合诊断)。V4 按[有界训练对照](DATASET_COVERAGE_TRAINING.md#4-v4-有界训练对照)分别比较缩量、等量覆盖、调度与增强。冻结实拍只用于入选版本的固定口径回归。

## 复跑入口

用与检查点一致的 Python 3.12 便携环境启动 [evaluate_field_capture.py](../../lab/scripts/data/evaluate_field_capture.py) 的 `predict`。传入 `--batch datasets/annotated/site_capture_20261112`、`--checkpoints datasets/annotated/site_capture_20261112/field_eval_v1/v3_checkpoints.json` 和新的 `--run runs/<名称>`。评分用 `score --run runs/<名称> --output runs/<名称>/score_conf025 --confidence 0.25 --labels datasets/annotated/site_capture_20261112/field_eval_v1/evaluation_candidates.json`。同一批推理文件可复用作阈值诊断，正式阈值在另一个验证集合确定。
