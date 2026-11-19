# 双针 Pose 读数与独立推理包

入口为 [infer_meter_pose.py](scripts/infer_meter_pose.py)，配置为
[pose_readout.json](config/pose_readout.json)。温度 −30～50°C、1°C 一小格，
湿度 0～100%RH、2%RH 一小格；两组分别使用各自的轴心、针尖、量程两端。
范围、小格、主刻度和顺逆时针方向均读取配置。

核心复用 `ws/src/aec_vision/src/aec_vision/meter.py` 的 `read_pose_dials`、
`equal_angle_scale` 和 `reading_from_ticks`。绘图中的绿色刻度是等角度推算，
没有检测或确认每条印刷刻度，也没有透视校正；结果为当前模型点位下的估算。
低置信度点、退化几何、空白弧和未检出分别记录 unknown，温湿度两项互不填充。
图片输出原图标注、放大预览与 JSON；视频输出逐帧标注 MP4、JSONL 与汇总。

从仓库根运行：

```powershell
.venv/Scripts/python.exe -B lab/features/meter/inference/scripts/infer_meter_pose.py --source "图片或视频路径"
.venv/Scripts/python.exe -B -m unittest lab.features.meter.inference.tests.test_pose_readout lab.features.meter.inference.tests.test_needle
```

按用户最新要求，使用专门的
[package_meter_inference.py](scripts/package_meter_inference.py) 生成轻量目录，
包含 Python 脚本、模型和配置，不附带解释器、虚拟环境或依赖安装包。
运行依赖目标电脑已有的 Python 与相关库，已测版本记录在包内 `requirements.txt`。
不包含训练数据，也不依赖相邻的训练包。

```powershell
.venv/Scripts/python.exe -B lab/features/meter/inference/scripts/package_meter_inference.py
```

产物为 `training_packages/meter_inference`，已有目录不覆盖。复制整个目录后执行
`python infer_meter.py --source 图片路径`，或 `run.bat --source 图片路径`。
默认 CUDA 0，CPU 使用 `--device cpu`；`run.ps1` 可用 `-Python` 指定已有解释器。
包内配置将模型路径改为 `models/best.pt`，核心和推理脚本作为源码快照一起交付；
修改原维护源后需要重新构建，既有包不会自动更新。

现有训练目录 `training_packages/meter` 也留有 `infer_meter.py`、核心快照、配置
和训练后 `models/best.pt`。独立外发使用 `meter_inference`，其
`package_manifest.json` 保存文件来源及摘要，`outputs/` 保存实际推理小样。
当前为 Windows 离线检查，研究结果不等于 ROS/实车验收。
