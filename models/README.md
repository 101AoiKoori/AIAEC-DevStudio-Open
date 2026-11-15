# 模型区

权重存放在忽略的 `weights/`。每个模型记录来源、训练数据版本、类别顺序、输入尺寸、运行时依赖与实车推理延迟。省赛模型不可直接视为国赛验收通过。模型权重、训练运行和部署模型的类归属与版本策略见 `config/workspace_ontology.yaml`。

此处用于训练与研究。正式识别模型文件直接部署到 `ws/src/aec_vision/models/`，不增加任务或模型子目录；多文件模型同样遵循这一约定，详见[识别开发设计](../docs/PERCEPTION_DEVELOPMENT.md#代码与数据的位置)。

`weights/plate/provincial/` 保存从省赛工程导入的车牌检测权重、完整七位 CNN 检查点与 PaddleOCR v4 推理模型，用于模型研究及后续统一训练。省赛原 CNN 仅保留作研究对照。正式 ROS 部署的车牌 CNN 使用 `legacy_control_20261106` 混合数据微调权重，仲裁为 CNN 优先。来源、输入尺寸、字符范围、版本区别与验证边界见[省赛车牌双 OCR](../docs/research/PLATE_DUAL_OCR.md)。这些权重不提交 Git，具体再分发许可待核实，实车时延未测。

`weights/e_bike/sam2_t.pt` 来自 [Ultralytics 模型资源](https://github.com/ultralytics/assets/releases/tag/v8.2.0)，用于电动车素材的两帧提示框分割对照，未用于正式推理。上游 [SAM 2](https://github.com/facebookresearch/sam2) 采用 Apache-2.0；本机对照通过 Ultralytics 8.2.103 实现，使用该实现须遵循其许可。输出掩码的轮辐和边缘仍需人工复核，目标机延迟未测。
