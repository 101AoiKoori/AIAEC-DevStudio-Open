# 数据集资源管理器截图（2026-11-20）

截图均为 Windows 资源管理器“中图标”视图，显示对应目录中当前排序靠前的文件。窗口状态栏的项目数只表示截图所打开目录内在位的文件数，不等于数据集总量或独立来源数。这些缩略图截图用于技术文档展示，不代替原始图像、标签和清单。

| 截图 | 实际打开目录 | 图注建议 |
| --- | --- | --- |
| [指针仪表](meter_train.png) | `AIAEC-DevStudio/datasets/exports/meter/train/images` | V4 指针仪表训练图像，目录 6480 项；可见白盘／彩盘与场景合成 |
| [楼名纸条](building_name_paper_train.png) | `AIAEC-DevStudio/datasets/exports/building_name_paper/train/images` | V4 楼名纸条训练图像，目录 3000 项；用于纸条定位 |
| [禁行牌](sign_train.png) | `AIAEC-DevStudio/datasets/exports/sign/train/images` | V4 禁行牌训练图像，目录 2500 项 |
| [物品贴纸](waste_sticker_train.png) | `AIAEC-DevStudio/datasets/exports/waste_sticker/train/images` | V4 物品贴纸训练图像，目录 1800 项 |
| [桶开闭](waste_bin_state_train.png) | `AIAEC-DevStudio/datasets/exports/waste_bin_state/train/images` | V4 真实桶主导微调输入，目录 804 项；不是原 3500 图基线 |
| [电动车基线](e_bike_train.png) | `AIAEC-DevStudio/datasets/exports/e_bike/train/images` | V4 电动车首轮冻结基线，目录 108 项 |
| [人物小样](person_sample.png) | `AIAEC-DevStudio/datasets/samples/person/images/synthetic` | 人物当前在位小样中的 52 张合成图；正式导出图像未在本机该目录完整在位 |
| [火焰小样](fire_sample.png) | `AIAEC-DevStudio/datasets/samples/fire/images/train` | 火焰当前在位小样的 Train 子目录 124 图；正式导出图像未在本机该目录完整在位 |
| [人物 11 月候选](person_nov20_candidate.png) | `AIAEC-VisualRecognition/person/datasets/v003_nov20_human_composite/images/train` | 相邻工作空间人物候选集 Train 目录 278 图，与国赛 V4 独立计数 |
| [电动车 11 月候选](e_bike_nov20_candidate.png) | `AIAEC-VisualRecognition/e_bike/datasets/v001_parking_row_review/images/train` | 相邻工作空间新增停车列训练候选 80 图；66 原图加 14 安全裁区 |
| [桶开闭 11 月候选](waste_bin_nov20_candidate.png) | `AIAEC-VisualRecognition/waste_bin_state/datasets/v001_four_bin_real_blender_joint/images/train` | 相邻工作空间实拍加 Blender 联合训练候选 478 图 |

图片只是文件夹外观证据，不显示标签、训练效果或图签审查通过。数据集规模及分集口径分别查 `docs/research/DATASET_MODEL_REPORT_20261119.md` 和相邻工作空间各任务 `dataset.json`。
