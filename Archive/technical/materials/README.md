# 素材统一入口

原始素材统一入口是 [datasets/raw](../../../datasets/README.md)，按任务和来源组织。下表为 2026-11-09 的目录文件数：

| 任务 | 本机位置及内容 | 来源和用途 |
| --- | --- | --- |
| 人物 | `datasets/raw/person/team_assets/`，19 个文件 | 16 张社区人物、2 张非社区人物图案及尺寸照片；比赛人数按现场布置确认 |
| 火焰 | `datasets/raw/fire/team_assets/`，2 个文件 | `image1.png` 是用户文档中的图案；`实拍.jpg` 是贴纸近照，纸面下缘截断 |
| 车牌 | `datasets/raw/plate/team_assets/` 和 `provincial_source_pic/`，共 8 个文件 | 三张队员素材与五张省赛合成素材 |
| 仪表 | `datasets/raw/meter/`，2 个文件 | PDF 提取图和外部参考图；外部来源许可及与实物一致性待核实 |
| 垃圾贴纸 | `datasets/raw/waste_bin/source/` 和 `toolkit/`，共 64 个文件 | 队员工具包压缩包及解压工作目录，含配置、脚本、说明和缓存 |
| 其余任务 | `datasets/raw/{e_bike,sign,thermal,traffic_light}/` | 核对时目录为空 |

国赛人物素材中的 18 张人物图案与相邻人物检测工程的同名文件逐字节一致；尺寸照片存于本工作空间。外来素材的许可及采集授权待核实。模型、标注数据和训练导出见[迭代对照](../iterations/README.md)。

垃圾贴纸历史数据位于 `datasets/exports/waste_bin/archive/`；研究模型权重位于 `models/weights/`。数据与模型不提交 Git。

[Archive/Assets 图片与视频逐文件索引](ASSETS_INDEX.md)记录楼宇、垃圾桶、电动车及场景参考的内容、视角与检查范围。素材原件按楼宇、垃圾桶、电动车与人物、场景、火焰纸片、仪表、红绿灯、车牌及禁止直行牌分类，分别置于 `Archive/Assets/` 对应目录；楼宇下按楼名和素材类型分别存放。其中 14 张小车摄像头实拍图片来自 `pics.zip`，文件名和索引已标注来源，可参考小车相机的真实取景；采集参数及与当前安装条件的一致性尚未确认。
