# 技术版本入口

这里收录有明确迭代边界的数据集、脚本、模型及技术配置。每个版本记录原路径、来源和状态；代码历史以 Git 为依据。

## YOLO 训练 V1/V2（2026-11-11）

本轮在国赛工作空间执行 18 项低预算试验；各模型 V1/V2 使用同一组 50 张训练图、320 输入和完整 val/test。V1 为 SGD、lr0 0.01、1 epoch，V2 为 SGD、lr0 0.005、2 epoch。真实结果与每任务比较图表见[训练流程](../../../docs/TRAINING_WORKFLOW.md#2026-11-11-冒烟试验结果)。

| 模型 | V1 | V2 |
| --- | --- | --- |
| 人物 | [person_yolo_v1](person/person_yolo_v1/) | [person_yolo_v2](person/person_yolo_v2/) |
| 电动车 | [e_bike_yolo_v1](e_bike/e_bike_yolo_v1/) | [e_bike_yolo_v2](e_bike/e_bike_yolo_v2/) |
| 火焰纸贴 | [fire_sticker_yolo_v1](fire/fire_sticker_yolo_v1/) | [fire_sticker_yolo_v2](fire/fire_sticker_yolo_v2/) |
| 楼名纸条 | [building_name_paper_yolo_v1](fire/building_name_paper_yolo_v1/) | [building_name_paper_yolo_v2](fire/building_name_paper_yolo_v2/) |
| 禁行牌 | [sign_yolo_v1](sign/sign_yolo_v1/) | [sign_yolo_v2](sign/sign_yolo_v2/) |
| 桶开闭 | [waste_bin_state_yolo_v1](waste_bin/waste_bin_state_yolo_v1/) | [waste_bin_state_yolo_v2](waste_bin/waste_bin_state_yolo_v2/) |
| 垃圾贴纸 | [waste_sticker_yolo_v1](waste_bin/waste_sticker_yolo_v1/) | [waste_sticker_yolo_v2](waste_bin/waste_sticker_yolo_v2/) |
| 仪表 | [meter_yolo_v1](meter/meter_yolo_v1/) | [meter_yolo_v2](meter/meter_yolo_v2/) |
| 车牌检测 | [plate_detection_yolo_v1](plate/plate_detection_yolo_v1/) | [plate_detection_yolo_v2](plate/plate_detection_yolo_v2/) |

每版保存冻结的 `scripts/`、`config/train.yaml` 和独立 `runs/`；权重、日志、TensorBoard、数据清单、逐图预测、验证／测试指标和样本图均在该版本的 `runs/smoke/`。代码、小配置和来源元数据纳入 Git，图片、权重和批量运行产物仅在本机归档。图表未生成项由独立 `postprocessing/` 清单与说明图记录，原始指标和图表保留。

原位置为 `runs/smoke_versions/training_iterations_20261111/<模型>_<版本>/`。源码在准备时冻结自 `lab/training_common/`；当时 Git HEAD 为 `f3bdd0aa2e135be2a09e12bd4ea1f49bde736351`，工作树含未提交训练工程，因此该提交号本身不代表快照全部内容。每版 `runs/provenance.json` 记录实际源路径与快照对应、准备时工作树状态，`runs/preparation_environment.json` 记录软件环境。当前 V3 的后续工程补丁不追写这些冻结脚本。

归档移动后核对 18 项完成状态、同任务训练样本一致性、类别覆盖、完整评估数量及运行产物；对应证据为原 suite 目录的 `suite.json`、`artifact_audit.json` 和 `git_snapshot_files.txt`。V3 开发源仍在各任务 `training/`；分发包镜像曾位于 `training_packages/`，已于 2026-11-15 删除，该目录现只保留 `portable_env/`。

## 三栋楼 Blender 模型

下表是国赛工作空间内产生的楼宇模型及对应预览。每个原路径为 `datasets/annotated/fire/blender_buildings_v版本号/`；相同版本号的归档目录与原文件名保持对应。各模型对应的具体脚本提交未确定；构建脚本的已追踪版本见 `lab/features/fire/training/scripts/build_blender_buildings.py` 的 Git 历史。

| 版本 | 本机归档位置 | Blender 模型数 | 预览图数 |
| --- | --- | ---: | ---: |
| v1 | [blender_buildings_v1](fire/blender_buildings_v1/) | 1 | 6 |
| v2 | [blender_buildings_v2](fire/blender_buildings_v2/) | 1 | 6 |
| v3 | [blender_buildings_v3](fire/blender_buildings_v3/) | 1 | 13 |
| v4 | [blender_buildings_v4](fire/blender_buildings_v4/) | 1 | 14 |
| v5 | [blender_buildings_v5](fire/blender_buildings_v5/) | 1 | 14 |
| v7 | [blender_buildings_v7](fire/blender_buildings_v7/) | 1 | 15 |
| v8 | [blender_buildings_v8](fire/blender_buildings_v8/) | 1 | 16 |
| v9 | [blender_buildings_v9](fire/blender_buildings_v9/) | 1 | 16 |
| v10 | [blender_buildings_v10](fire/blender_buildings_v10/) | 4 | 16 |
| v11 | [blender_buildings_v11](fire/blender_buildings_v11/) | 4 | 16 |
| v12 | [blender_buildings_v12](fire/blender_buildings_v12/) | 4 | 16 |

v6 模型及预览缺失，生成状态未知；当前 v13 保留在数据区。v2–v12 原目录的字体和许可与 v13 同名文件逐字节一致，重复副本已删除；v1 的不同字体及许可保留原位，v13 字体及许可保留。模型内打包 Noto Sans SC 字体与用户提供的火焰图案，其来源与许可边界见[数据区说明](../../../datasets/README.md)。这些模型和预览是渲染派生物；实拍提供的楼体外观线索沿原始照片/视频来源记录，渲染产物不计为新的相机拍摄，也不提交 Git。

## 垃圾桶 Blender 模型

下表是国赛工作空间内产生的垃圾桶模型及对应预览。每个版本的原位置为 `datasets/annotated/waste_bin/` 下与归档目录同名的目录；具体脚本提交对应关系未确定，构建脚本历史见 `lab/features/waste_bin/training/scripts/build_blender_waste_bin.py` 的 Git 记录。

| 本机归档位置 | Blender 模型数 | 预览图数 |
| --- | ---: | ---: |
| [blender_reference_20261109](waste_bin/blender_reference_20261109/) | 1 | 1 |
| [blender_reference_v2_20261109](waste_bin/blender_reference_v2_20261109/) | 1 | 1 |
| [blender_reference_v3_20261109](waste_bin/blender_reference_v3_20261109/) | 1 | 1 |
| [blender_reference_v4_20261109](waste_bin/blender_reference_v4_20261109/) | 1 | 1 |
| [blender_reference_v5_20261109](waste_bin/blender_reference_v5_20261109/) | 1 | 2 |
| [blender_reference_v6_20261109](waste_bin/blender_reference_v6_20261109/) | 1 | 2 |
| [blender_reference_v7_20261109](waste_bin/blender_reference_v7_20261109/) | 1 | 2 |
| [blender_reference_v8_20261109](waste_bin/blender_reference_v8_20261109/) | 1 | 2 |
| [blender_reference_v10_20261109](waste_bin/blender_reference_v10_20261109/) | 1 | 2 |
| [blender_reference_v11_20261109](waste_bin/blender_reference_v11_20261109/) | 1 | 2 |
| [blender_reference_v12_20261109](waste_bin/blender_reference_v12_20261109/) | 1 | 2 |
| [blender_pbr_v1_20261110](waste_bin/blender_pbr_v1_20261110/) | 1 | 1 |

`blender_reference_v9_20261109` 原目录为空，模型和预览缺失。当前 PBR v2 保留在数据区；PBR v3 仍待核。v13 旧构建基线及预览已归档见[历史基线记录](waste_bin/waste_bin_reference_v13_20261109/)。其场景由 2026-11-09 的三个现存批次 manifest 和一个归档批次 manifest 声明为 `scene_file`；这些是生成时来源声明，不确立其与 PBR v2 的精确父子关系。场景包含内嵌 Wikimedia 标识，署名及 CC BY-SA 4.0 许可见[标识署名文件](../../../datasets/annotated/waste_bin/wikimedia_decals_20261109/attribution.json)。归档模型和预览是本机文件，不提交 Git。

### 垃圾桶 v13 历史构建基线

[归档文件夹](waste_bin/waste_bin_reference_v13_20261109/)保留 `waste_bin_reference.blend`、`preview.png` 和 `preview_detail.png`。依据实拍和用户测量近似建模；场景深度及车载观察距离未测。Blender 5.2.3 成功打开模型：87 个对象，4 张 768×768 标识图均已打包，无外部库或缺失图像。当前文本检索未发现活动生成入口消费；四份历史 manifest 记录其来源。旧清单中的 `scene_file` 路径保留为生成时原始声明。场景和预览不由 Git 恢复；标识来源及署名见[标识署名文件](../../../datasets/annotated/waste_bin/wikimedia_decals_20261109/attribution.json)。

## 仪表 Blender 模型

[blender_9011_pbr_v1_20261110](meter/blender_9011_pbr_v1_20261110/) 包含一个 PBR 模型及一张预览，原位置为 `datasets/annotated/meter/blender_9011_pbr_v1_20261110/`。模型打包了生成式表盘贴图，贴图许可及与实物的一致性待核；模型对应的具体脚本 Git 提交未确定。当前 PBR v2、可编辑参考模型和贴图保留在数据区；旧 `.blend1` 静态备份已归档为恢复候选，直接版本关系待核。归档模型与预览不提交 Git。

| 分支状态 | 原路径 | 归档位置 | 文件与依据 |
| --- | --- | --- | --- |
| 参数化站房初始父偏移分支，已退役 | `datasets/annotated/meter/v4_parametric_station_20261113/` | [parametric_station_parent_offset_retired_20261113](meter/parametric_station_parent_offset_retired_20261113/) | 1 个 `station.blend`、`face.png`、`endpoints.json`。归档文件来自当前磁盘快照；未找到可恢复这些二进制的 Git 提交。`endpoints.json` 的 `index_lifecycle.status` 为 `retired`，依据是初始 face 与 markers 使用错误偏移。关联 pilot 原路径 `datasets/exports/meter/v4_parametric_station_pilot_20261113/` 的归档映射见[数据集版本索引](../dataset-versions.tsv)；其 manifest 记录 `corrected v2 supersedes this pilot`，生成时的 `source_blend` 仍保留原路径。退役原因不证明仪表实物刻度对应、端点可见性或 Pose 真值已验证。 |
| 自动备份静态恢复候选，直接谱系待核 | `datasets/assets/blender/meter/blender_9011_reference/meter_9011_approx.blend1` | [meter_9011_approx_static_backup_candidate_20261106](meter/meter_9011_approx_static_backup_candidate_20261106/) | 保留原 `.blend1` 文件格式。Blender 5.2.3 打开归档文件成功：13 个对象，512×512，帧终点 250，无对象动画；相机 Z 为 4.2，两根指针为零旋转。当前同名 `.blend` 为 1024×1024、帧终点 721，相机 Z 为 3.944，指针和相机各有 121 个关键帧；两文件对象名、网格计数、材质和 1254×1254 打包表盘贴图一致，均无外部库。源文件修改时间早于当前 `.blend`，且代码/文档未发现对 `.blend1` 的活动引用。该备份保留了相较当前文件不同的静态状态，故归档作本机恢复候选；它是否为当前动画模型的直接前一版、动画是否为两者唯一变化，均未证实。该归档二进制不在 Git 中。 |


## 共享场景候选备份

| 备份及原路径 | 归档位置 | 对照记录 |
| --- | --- | --- |
| `datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_4_2m.blend1` | [arena_4_2m_backup_candidate_20261112](arena/arena_4_2m_backup_candidate_20261112/) | Blender 5.2.3 可打开。当前 `.blend` 与备份均有 11 个对象、7 个集合、5 个材质、640×480 输出；逐项结构检查中的对象名、类型、变换、尺寸、网格计数、材质名、相机和依赖结构一致。两者均无文件图像依赖或外部库。未做渲染比较；修改时间相差约 24 秒，不证明完整等价或直接保存谱系。归档作本机恢复候选。 |
| `datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_indoor_calibrated.blend1` | [arena_indoor_calibrated_backup_candidate_20261112](arena/arena_indoor_calibrated_backup_candidate_20261112/) | Blender 5.2.3 可打开。备份 18 个对象，当前 `.blend` 20 个；备份墙高 3m，缺少 `Reference crossing line` 和 `Reference side line`，`Vehicle camera` 位置为 `(-0.6,-1.45,0.38)`，当前为 `(-0.6,-0.75,0.42)`。材质名称和 640×480 输出相同；无外部库，图像依赖均可解析。未发现 `.blend1` 活动消费者。作为几何/机位恢复候选保存；它是否为当前校准场景的直接前一版未知。 |
| `datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_indoor_detailed.blend1` | [arena_indoor_detailed_material_backup_candidate_20261112](arena/arena_indoor_detailed_material_backup_candidate_20261112/) | Blender 5.2.3 可打开。对象、位置、尺寸、网格结构、材质名称和 640×480 输出相同；材质 `PBR asphalt charcoal` 的 Base Color 当前为 `(0.002,0.002,0.002,1)`、备份为 `(0.003,0.003,0.003,1)`，Specular IOR Level 当前为 `0.12`、备份为 `0.23`，其他抽查的节点输入和连接相同。无外部库，图像依赖均可解析；未发现 `.blend1` 活动消费者。未进行渲染对比，直接保存谱系未知，归档作材质参数恢复候选。当前 `arena_indoor_detailed.blend` 仍被人物、禁行牌和车牌生成入口引用；其记录的父场景仍是当前 `arena_indoor_calibrated.blend`，不指向此备份。

三份 arena 文件均保留 `.blend1` 格式，归档二进制在本机存在且不由 Git 恢复。SQLite 保留原路径别名，并以 `backup_candidate_for` 记录与当前 `.blend` 的候选关系；这不确立直接版本谱系。
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
