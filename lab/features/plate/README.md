> 自动生成：请修改 `config/perception_tasks.json`，然后运行 `python lab/scripts/contracts/sync_docs.py`。

**识别摘要：** 1～3 号车位的车牌字符 [16]

**分值：** 12

**调度方案：** 后台处理（待接入调度）。按车位定点正对拍照，后台绑定车位并运行既定字符识别与第二路 OCR。

[开源项目分析与方法研究](../../../docs/research/PLATE_DUAL_OCR.md)

# 1～3 号停车位车牌识别

规则：第 16、23、25 页；三个车位每处 4 分。示例：“1 号停车场车牌号为 苏AN325J。”

采样与处理按[定点正对采样](../../../docs/PROJECT_DIRECTION.md#定点正对采样)执行。

事项标签见[需求状态约定](../../../docs/PERCEPTION_DEVELOPMENT.md#需求状态与提醒)。

## 需要实现

- `[待解决]` PLATE-R01：在各车位任务点正对拍照，输入车牌画面与车位编号，检测车牌位置并识别完整字符，输出每个车位各自的号码。
- `[已解决]` PLATE-R02：当前核心仲裁已在 `ws/src/aec_vision/src/aec_vision/plate.py` 实现，解决范围为双路文本决策；车端定点观察集成另行验收。比赛使用混合数据微调后的七位置 CNN，PaddleOCR 作为同图第二识别器。仲裁规则：两路合法文本相同则按 `agreement` 确认；两路均合法但不同时，采用微调 CNN 文本，记录 `cnn_primary_disagreement` 并原样保留 Paddle 文本与两路分数；CNN 文本格式无效时，仅当合法 Paddle 文本自身分数达到 0.70 才回退采用 Paddle，否则输出未确认。模型分数只分别记录，不跨模型比较。
- `[待解决]` PLATE-R03：每个车位终端输出并播报号码，在输出图片中框选车牌。编号来源于观察任务或现场可识别标志，需要与画面准确绑定。
- `[待解决]` PLATE-R04：单次照片无可靠结果要标记未识别；不可将一个车位的结果用于另一个车位。


## 工作区与验收

`ws/src/aec_vision/src/aec_vision/plate.py` 放检测和字符识别实验，`inference/config/` 记录字符集及车位信息，`inference/tests/` 评估。已有模型及来源对照按指定的回归或现场验收范围使用。分别检查检测、完整车牌准确率、车位归属、框图及播报；国赛现场验收按实际运行结果记录。

训练、实验目录与交接约定见 [识别开发设计](../../../docs/PERCEPTION_DEVELOPMENT.md)。
## 状态与提醒

- `[已明确]` PLATE-DATASET-SCOPE：V4 车牌使用省赛模型，数据集制作范围为[八项数据任务](../../../datasets/DATASET_COORDINATION.md#工作对象和落点)。代码、模型及车位关联的适配状态按研究和运行记录核对。
- `[已解决]` PLATE-TRAINING-V3：plate_detection 的 V3 训练入口和配置见本任务 `training/`。历史运行及效果见[V3 回传记录](../../../docs/research/FIELD_EVALUATION_V3_20261112.md#v3-训练行为)。

- `[已明确]` PLATE-1：双 OCR 角色。自研 CNN 为主识别器，PaddleOCR 为当前第二路。更换第二路时按 PLATE-2 核对契约。
- `[待解决]` PLATE-2：替换契约；第二路输出引擎身份、文本、独立分数、状态及版本；替换后重新确定自身门槛，不沿用 Paddle 的 0.70。
- `[待全局规划]` PLATE-3：定点观察与算力；协调各车位正对观察点和楼名/车牌 OCR 的模型驻留；每个车位使用该点正对拍摄的照片。
