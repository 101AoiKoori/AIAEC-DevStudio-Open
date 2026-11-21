# 楼名纸条的停点定位与文字读取边界

**设计分类：载体与空间归属。** 停点楼面图 → 纸条框 → 文字转录 → 对应楼体。纸条可定位、字符可读和楼体可归属分别记录。

**验证问题。** 在斜视、蓝车顶及空背景干扰下验证纸条检测，并量化从广场景覆盖转向固定停点的得失。


## 定义与作用范围

楼名纸条检测以小车在指定任务点面向楼面所拍的近距原图为输入，定位可见纸条，输出单类 `paper` 的框。VisualRecognition 当前验收看常规任务点的检出、同纸重框及误检，路线远景归入历史，大旋转与明显截边用于压力测试。DevStudio 数据契约可另带可读文字真值，分别记录纸条位置、文字和纸张实例。后续 OCR 及跨帧唯一计数各有独立验收。查 [DevStudio 楼名数据定义](../../datasets/scripts/building_name_paper/README.md) 与 VisualRecognition 的 `building_name_paper/task.yaml`。

## 正例与反例

| 观察 | 处理 |
| --- | --- |
| 正对任务点看到楼面上的纸条，纸面和边界清楚 | 标一个 `paper` 框，有可信全文时另记录文字 |
| 斜视下纸面仍能定位，但字已不可读 | 可计定位正例，文字记未知 |
| 同一纸条出现两个重叠预测 | 标注仍只有一张，第二个匹配算重框 |
| 白窗格、蓝车顶、墙扇或背包，楼面可见处没有纸条 | 可作为误报对照，已确认全图任务区无纸条时使用空标签 |

## 核心问题与可观测证据

一张贴在墙上的纸条由纸边、印刷行和附着位置共同提供证据。相机离轴会改变纸面投影，实纸弯翘时边界、字行、反光和接触阴影也随一起变化。平面透视只处理共面几何。格式塔的图底和邻近分组可解释布局，例如把同一纸面的多行文字归为一个实例，避免按每行文字或多段纸边各报一框。窗格与车顶可有相似的矩形边缘，仍需从纸面和附着关系判断。 [Wagemans 等，2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3482144/)

## 数据与方法设计

- 实拍保留纸张纤维、高光、翘边、贴附、低机位和镜头模糊。近点正面、斜视、无纸楼面与同框干扰分别覆盖。按父视频和场次分组，相关邻帧作为同源对照。
- Python 的实拍底图替字可在已定位的纸面上低成本改变字形、文字和曝光，保留正常字序。Blender 可按楼体、机位和纸张掩膜渲染有纸／无纸图层，物理接地阴影与实拍底图混合。最终框与最终可见纸面一致。若需补弯折设计，可参考 DocUNet 对非平面文档变形的建模，实际方法需与实拍纸材核对。 [Ma 等，2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Ma_DocUNet_Document_Image_CVPR_2018_paper.html) · [V4 制作入口](../../datasets/scripts/building_name_paper/README.md)
- 对容易错检的物体，历史实验使用少量图生图补车顶、墙扇等难负和纸条同框正例，逐图核边。文本合成文献可给 OCR 专项提供字样和成像变化的参考，检测与文字监督分别出具证据。 [Jaderberg 等，2014](https://arxiv.org/abs/1406.2227)

## 历史与当前

**DevStudio 历史。** V4 用实拍、实拍底图替字、完整 Blender、楼体加实拍背景编排检测图签，正式检测集合为 3,000／300／300。视频微调在 21 张新同场景 Val 上整图全对从 10 增到 18，旧 Test 在同一 0.25 阈值下却从 268/300 降到 264/300。这批结果只给纸条检测对照。 [V4 阶段报告](../research/DATASET_MODEL_REPORT_20261119.md)

**VisualRecognition 当前（2026-11-21）。** 早期试过无纸难负、斜视正样本、纸条与蓝车顶同框、旧样本回放以及冻结基线加六块裁图。用户明确只在面前停点识别后，选用 v004 同框微调第 41 轮快照 `v001_vehicle_selected`，整图一次 640 推理，置信度 0.70，NMS IoU 0.70。四次常规通行合计 20 张车载图中 12 个纸条均检出，8 张停车点无误报，另有 3 个固定点均通过重框和误报复测。证据见 VisualRecognition 的 `building_name_paper/STATE.md`、`building_name_paper/models/v001_vehicle_selected/model.json` 和 `building_name_paper/experiments/v004_real_paper_finetune/r007_selected_snapshot_check/run.json`。

## 障碍、取舍与验证

路过时的微小目标会让裁块成为有意义的工程候选。历史探索用六个重叠裁块在一张图补回两个远处纸条，但正式 Val 选出的高分支阈值在独立车载图中没有新增检出，降低阈值还会带来额外误框。固定停点近拍使单次整图识别更贴合现场运行预算，节省裁块合并、阈值调试和额外推理。v004 在旧 Test 的宽松匹配召回与框精度存在取舍，0.70 时严格 IoU 0.90 的检出从 274/334 降到 174/334，旧样本回放对照说明框精度值得单列。定位验收分别记录近距检出、同纸重框、同框误报和框精度。涉及文字或逐楼计数时，另以字符真值和观察覆盖验证。

## 实验结果与证据范围

| 对照与样本范围 | 已记录结果 | 结论与采用状态 |
| --- | --- | --- |
| DevStudio，21 张新同场景 Val 与 300 张旧 Test，conf 0.25 | 整图全对分别由 10/21→18/21、268/300→264/300 | 微调在新机位获益，旧分集整图结果下降。 |
| VisualRecognition，四次通行 20 张原图，640 输入，conf 0.70 | 12 个纸条均检出，8 张停车负图均无误报 | 现选停点方案。这组数据只覆盖纸条检测，转录和楼名归属各需相应证据。 |

## 实现与复核路径

- **任务契约。** 从识别需求定位业务输出，再按学习目标核对监督字段及有效范围，按组合轴查相应对照条件。[识别需求](../../config/perception_tasks.json) · [学习目标](../../config/task_targets.yaml) · [组合轴](../../config/combination_axes.yaml)
- **数据到模型。** 从该任务的来源配置追踪当前输入，由任务脚本生成图签及清单，再由训练入口读取数据。试验所用快照以当次记录为准。[数据来源配置](../../datasets/scripts/building_name_paper/config/source.yaml) · [任务制作说明](../../datasets/scripts/building_name_paper/README.md) · [V4 训练入口](../../lab/features/fire/training/scripts/train_building_name_paper_v4.py)
- **结果复核。** DevStudio 的数值回查对应评估分集和阶段报告。VisualRecognition 的本地记录按本篇的任务状态与实验路径复查。[V4 阶段报告](../research/DATASET_MODEL_REPORT_20261119.md) · [跨工作区记录口径](README.md#资料与时间范围)
- **VisualRecognition 记录。** `building_name_paper/models/v001_vehicle_selected/model.json` 固定选用快照和推理口径，`building_name_paper/experiments/v004_real_paper_finetune/r007_selected_snapshot_check/run.json` 记录停点复测。
