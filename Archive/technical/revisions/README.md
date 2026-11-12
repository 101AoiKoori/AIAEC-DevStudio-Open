# 技术版本入口

> 本文提及的部分本地素材、数据和历史运行产物未随本仓库提供；相关名称保留用于说明来源和实验背景。

这里收录有明确迭代边界的数据集、脚本、模型及技术配置。每个版本记录原路径、来源和状态；代码历史以 Git 为依据。

## YOLO 训练 V1/V2（2026-11-11）

本轮在国赛工作空间执行 18 项低预算试验；各模型 V1/V2 使用同一组 50 张训练图、320 输入和完整 val/test。V1 为 SGD、lr0 0.01、1 epoch，V2 为 SGD、lr0 0.005、2 epoch。真实结果与每任务比较图表见[训练流程](../../../docs/TRAINING_WORKFLOW.md#2026-11-11-冒烟试验结果)。

| 模型 | V1 | V2 |
| --- | --- | --- |
| 人物 | person_yolo_v1 | person_yolo_v2 |
| 电动车 | e_bike_yolo_v1 | e_bike_yolo_v2 |
| 火焰纸贴 | fire_sticker_yolo_v1 | fire_sticker_yolo_v2 |
| 楼名纸条 | building_name_paper_yolo_v1 | building_name_paper_yolo_v2 |
| 禁行牌 | sign_yolo_v1 | sign_yolo_v2 |
| 桶开闭 | waste_bin_state_yolo_v1 | waste_bin_state_yolo_v2 |
| 垃圾贴纸 | waste_sticker_yolo_v1 | waste_sticker_yolo_v2 |
| 仪表 | meter_yolo_v1 | meter_yolo_v2 |
| 车牌检测 | plate_detection_yolo_v1 | plate_detection_yolo_v2 |

每版保存冻结的 `scripts/`、`config/train.yaml` 和独立 `runs/`；权重、日志、TensorBoard、数据清单、逐图预测、验证／测试指标和样本图均在该版本的 `runs/smoke/`。代码、小配置和来源元数据纳入 Git，图片、权重和批量运行产物仅在本机归档。图表未生成项由独立 `postprocessing/` 清单与说明图记录，原始指标和图表保留。

原位置为 `runs/smoke_versions/training_iterations_20261111/<模型>_<版本>/`。源码在准备时冻结自 `lab/training_common/`；当时 Git HEAD 为 `f3bdd0aa2e135be2a09e12bd4ea1f49bde736351`，工作树含未提交训练工程，因此该提交号本身不代表快照全部内容。每版 `runs/provenance.json` 记录实际源路径与快照对应、准备时工作树状态，`runs/preparation_environment.json` 记录软件环境。当前 V3 的后续工程补丁不追写这些冻结脚本。

归档移动后核对 18 项完成状态、同任务训练样本一致性、类别覆盖、完整评估数量及运行产物；对应证据为原 suite 目录的 `suite.json`、`artifact_audit.json` 和 `git_snapshot_files.txt`。V3 开发源仍在各任务 `training/`，分发镜像位于额外的 `training_packages/`。

## 三栋楼 Blender 模型

下表是国赛工作空间内产生的楼宇模型及对应预览。每个原路径为 `datasets/annotated/fire/blender_buildings_v版本号/`；相同版本号的归档目录与原文件名保持对应。各模型对应的具体脚本提交未确定；构建脚本的已追踪版本见 `lab/features/fire/training/scripts/build_blender_buildings.py` 的 Git 历史。

| 版本 | 本机归档位置 | Blender 模型数 | 预览图数 |
| --- | --- | ---: | ---: |
| v1 | blender_buildings_v1 | 1 | 6 |
| v2 | blender_buildings_v2 | 1 | 6 |
| v3 | blender_buildings_v3 | 1 | 13 |
| v4 | blender_buildings_v4 | 1 | 14 |
| v5 | blender_buildings_v5 | 1 | 14 |
| v7 | blender_buildings_v7 | 1 | 15 |
| v8 | blender_buildings_v8 | 1 | 16 |
| v9 | blender_buildings_v9 | 1 | 16 |
| v10 | blender_buildings_v10 | 4 | 16 |
| v11 | blender_buildings_v11 | 4 | 16 |
| v12 | blender_buildings_v12 | 4 | 16 |

v6 原目录仅发现字体和许可，模型及预览缺失，生成状态未知；当前 v13 保留在数据区。各原版本目录中的 `NotoSansSC.ttf` 和 `OFL.txt` 保留原位；模型内打包 Noto Sans SC 字体与用户提供的火焰图案，其来源与许可边界见[数据区说明](../../../datasets/README.md)。这些模型和预览是渲染派生物；实拍提供的楼体外观线索沿原始照片/视频来源记录，渲染产物不计为新的相机拍摄，也不提交 Git。

## 垃圾桶 Blender 模型

下表是国赛工作空间内产生的垃圾桶模型及对应预览。每个版本的原位置为 `datasets/annotated/waste_bin/` 下与归档目录同名的目录；具体脚本提交对应关系未确定，构建脚本历史见 `lab/features/waste_bin/training/scripts/build_blender_waste_bin.py` 的 Git 记录。

| 本机归档位置 | Blender 模型数 | 预览图数 |
| --- | ---: | ---: |
| blender_reference_20261109 | 1 | 1 |
| blender_reference_v2_20261109 | 1 | 1 |
| blender_reference_v3_20261109 | 1 | 1 |
| blender_reference_v4_20261109 | 1 | 1 |
| blender_reference_v5_20261109 | 1 | 2 |
| blender_reference_v6_20261109 | 1 | 2 |
| blender_reference_v7_20261109 | 1 | 2 |
| blender_reference_v8_20261109 | 1 | 2 |
| blender_reference_v10_20261109 | 1 | 2 |
| blender_reference_v11_20261109 | 1 | 2 |
| blender_reference_v12_20261109 | 1 | 2 |
| blender_pbr_v1_20261110 | 1 | 1 |

`blender_reference_v9_20261109` 原目录为空，模型和预览缺失；当前参考 v13 与 PBR v2 保留在数据区，PBR v3 也保留原位，其使用状态待核。部分模型打包了 Wikimedia 派生标识，作者及 CC BY-SA 4.0 许可见标识署名文件。归档文件是本机模型和预览，不提交 Git。

## 仪表 Blender 模型

blender_9011_pbr_v1_20261110 包含一个 PBR 模型及一张预览，原位置为 `datasets/annotated/meter/blender_9011_pbr_v1_20261110/`。模型打包了生成式表盘贴图，贴图许可及与实物的一致性待核；模型对应的具体脚本 Git 提交未确定。当前 PBR v2、可编辑参考模型、贴图和 `.blend1` 备份保留在数据区；归档模型与预览不提交 Git。

## 垃圾贴纸生成脚本

已复原的垃圾贴纸生成脚本：

| 版本依据 | 日期 | 原路径 | 状态 |
| --- | --- | --- | --- |
| `5fcaea48a9cee7aece0335cf72d4ef083c9bd10d` | 2026-11-06 | `lab/features/waste_bin/training/scripts/generate_stickers.py` | 已复原并与 Git 内容逐字节核对 |
| `7fa3a93406b602fbb9686c4c03c609bae1078aaa` | 2026-11-06 | 同上 | 已复原并与 Git 内容逐字节核对 |
| `9fed2bc08a13363198ee0a4d8908b311c9a87619` | 2026-11-06 | 同上 | 已复原并与 Git 内容逐字节核对 |
| `dc188a24064759f77a8cb385719e2e6be2cb8416` | 2026-11-06 | 同上 | 已复原并与 Git 内容逐字节核对 |

技术文件按 `Archive/technical/revisions/<任务>/<对象及版本号>/` 分类；技术变化见[迭代对照](../iterations/README.md)。[脚本版本索引](../script-versions.tsv)与[技术配置索引](../config-versions.tsv)记录原路径、提交和归档位置，[相邻版本差异索引](../script-differences.tsv)记录前后提交和变化行数。已复原 28 个 Python 文件的 92 个版本及 4 个技术配置的 12 个版本，包含识别实验、ROS 包和工作空间维护脚本；版本号按同一路径的内容变化顺序确定。

车牌核心由 `ws/src/competition_perception/src/competition_perception/plate.py` 在提交 `1e133f1e022dbddaac385e2c1646dc4e6935e99f` 原样改名为 `ws/src/aec_vision/src/aec_vision/plate.py`（Git 识别为 R100）。两条路径分别列在脚本版本索引中。
