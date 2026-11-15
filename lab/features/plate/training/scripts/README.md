# 车牌场景脚本

`inspect_arena_plate.py` 检查车牌场景副本。`render_plate_arena_pilot.py` 和 `render_plate_arena_measured_batch.py` 在场景副本中拟合现场车体裁片；车体尺寸、位置和遮挡状态未测量。

`render_plate_empty_scene.py` 在空场景创建 36 × 24 cm 牌面和示意支架，使用现场帧 46 × 18 像素裁片贴图与 Cycles OPTIX 渲染。牌面贴图的比例发生变化；历史批次曾输出 108 张，来源清单 `datasets/exports/plate/v4_empty_scene_gpu_20261113/manifest.json` 已随导出目录清空。当前 V4 不生产车牌数据集。
