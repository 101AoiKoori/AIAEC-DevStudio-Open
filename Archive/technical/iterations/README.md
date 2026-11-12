# 迭代对照

按技术主题记录前后版本、原文件路径、变化内容和实验结果。不同主题可共用提交，一个主题可跨多个提交。

## 垃圾贴纸生成脚本

比较对象均为 `lab/features/waste_bin/training/scripts/generate_stickers.py`，四个版本的来源与复原状态见[版本入口](../revisions/README.md)：

| 前一版本 → 后一版本 | 变化 |
| --- | --- |
| `5fcaea4` → `7fa3a93` | 缓存位置改为通过参数指定 |
| `7fa3a93` → `9fed2bc` | 调整数据生成与核验流程，新增 134 行、删除 72 行 |
| `9fed2bc` → `dc188a2` | 调整脚本引用至工作空间内，新增 8 行、删除 1 行 |

可用 `git diff <前一提交> <后一提交> -- lab/features/waste_bin/training/scripts/generate_stickers.py` 逐行查看。

## 垃圾贴纸数据版本

`synthetic_train_v1`、`synthetic_train_v2`、`stickers_synthetic_v3` 和 `synthetic_smoke_20261106` 位于 `datasets/exports/waste_bin/archive/`，`stickers_synthetic_v4` 位于 `datasets/exports/waste_bin/`；[数据集版本索引](../dataset-versions.tsv)记录归档位置。v1、v3 中途停止，缺少完整清单。v4 为当前选用版本，训练集和调试验证集使用相同来源的图标。

## 仪表合成数据

仪表数据包括 PDF 单图增强试验、自绘流程试验、Blender 表盘与动画渲染候选、基于渲染底图的 OpenCV 增强数据。`datasets/annotated/meter/rendered_base_v1/` 有 89 张渲染底图，`datasets/exports/meter/opencv_rendered_meter_v1/` 有 1,000 张同源候选图，动画同一轨迹归入训练集。这两组见[数据集版本索引](../dataset-versions.tsv)；各流程试验的当前状态见[数据区](../../../datasets/README.md)。

## 车牌实验数据

本工作空间的车牌 CNN 对照使用 CCPD 与省赛合成数据。训练使用 CCPD 10,000 张和合成数据 5,580 张；验证分别使用 1,800 张和 620 张。测试使用 CCPD 2,700 张和从测试图检测裁剪的 500 张。数据来源见[车牌研究](../../../docs/research/PLATE_DUAL_OCR.md)。

## 车牌 CNN 权重对照

| 模型 | 实际本机来源 | CCPD test 2,700 张 | 检测裁剪 test 500 张 |
| --- | --- | ---: | ---: |
| 省赛 legacy 基线 | 当时记录的部署权重路径已被后续模型使用，测试文件与现存完整检查点的对应关系待核实 | 56.59% | 25.60% |
| 混合数据微调 legacy，epoch 9 | `lab/features/plate/training/runs/legacy_control_20261106/best.pt` | 83.33% | 47.40% |
| 混合数据微调 spatial_v2，epoch 10 | `lab/features/plate/training/runs/spatial_v2_20261106/best.pt` | 83.11% | 47.60% |

三组数值来自 `lab/features/plate/inference/outputs/cnn_comparison_20261106/comparison.json`，单位为整牌正确率。两组微调模型采用同一训练条件，分别使用 legacy 和 spatial_v2 结构。检测裁剪来自 CCPD 测试图。省赛基线记录的测试文件大小与现存完整检查点不同，二者对应关系待核实。国赛训练权重、元数据、逐轮指标和对照结果位于 `Archive/technical/assets/车牌/`。

历史脚本的逐路径版本与相邻差异见[版本入口](../revisions/README.md)，包括识别实验、ROS 包和工作空间维护 Python 文件。
