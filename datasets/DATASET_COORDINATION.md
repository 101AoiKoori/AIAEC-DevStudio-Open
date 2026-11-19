# V4 数据集制作协作协议

本文件用于查找任务负责人、脚本归属和协调共享修改。任务进度见[当前生产进度](../docs/DATASET_PRODUCTION.md#当前生产进度)，任务制作使用[数据制作 Skill](../.agents/skills/dataset-production-loop/SKILL.md)，跨窗口工作使用[协作 Skill](../.agents/skills/workspace-collaboration/SKILL.md)。

## 工作对象和落点

每个任务的数据脚本和配置位于 `datasets/scripts/<任务>/`。下表列出关联的开发区。

| 数据任务 | 当前开发职责位置 | 注意 |
| --- | --- | --- |
| `person` | `lab/features/person/` | 人物与电动车可共用来源，人物模型的标签单独维护。 |
| `e_bike` | `lab/features/e_bike/` | 姿态及计数以任务需求为准。 |
| `fire` | `lab/features/fire/` | 检测类别与模型键为 `fire_sticker`；与楼名任务协调共用楼面素材。 |
| `building_name_paper` | `lab/features/building_name_paper/` | 纸条检测及 OCR 的数据职责按任务需求确定。 |
| `sign` | `lab/features/sign/` | 牌具类别和标注范围以任务需求为准。 |
| `waste_bin_state` | `lab/features/waste_bin/` | 数据职责名与开发目录名不同；与垃圾贴纸共用桶素材。 |
| `waste_sticker` | `lab/features/waste_bin/training/` | 训练职责保留原位置；当前生成器独立位于 `datasets/scripts/waste_sticker/`。固定 24 图案，桶身标志与物品贴纸分别处理。 |
| `meter` | `lab/features/meter/` | 仪表定位和关键点/几何读数按当前需求与专项设计处理。 |

车牌沿用省赛模型。样本和正式集合分别写入 `datasets/samples/<任务>/`、`datasets/exports/<任务>/{train,val,test}/`；这些目录也是后续 agent 的明确落点，人物样本和 split 用 `.gitkeep` 保留空目录。样本或导出结果有误时，清空相应任务目录中的生成结果并重跑，保留目录及人工内容。任务命令见[脚本入口](scripts/README.md)。共用贝叶斯、透视、图像处理和 Blender 能力在 `datasets/scripts/common/`。本机有可用 GPU 时优先使用，必要的 CPU 运行或设备故障记入该次记录。

### 当前参与窗口

下表保留 2026-11-15 的本地窗口联系记录。需要联系时按用户当次授权核对 thread、host、项目与工作树。参与状态以当次实际连接为准。

| 职责 | 聊天名称 | thread ID |
| --- | --- | --- |
| 总体方向 | 启动 V4 数据集协作结构 | `01a1d807-5da4-7870-9774-166d804d1eba` |
| 跨窗口协调 | V4 · 协调镜像 | `01a1d83a-a59d-7ae3-b6a0-f0fab8d58021` |
| 全局文档治理 | AIAEC · 全局文档整治 | `01a1d83a-a7d5-7f22-ab61-437e80c7e80b` |
| V4 共享契约 | V4 · 文档与共享契约支持 | `01a1d808-5392-7100-a946-06f002bd9e85` |
| 资产与生命周期 | V4 · 资产与生命周期支持 | `01a1d808-560a-7081-886c-b9b42ba840bb` |
| 人物 | V4 · 人物数据集（新） | `01a1d89f-8984-7d30-aef7-22db083b00f4` |
| 电动车 | V4 · 电动车数据集（新） | `01a1d89f-8beb-7de1-b4b6-6ef88281526f` |
| 火焰贴纸 | V4 · 火焰贴纸数据集（新） | `01a1d89f-8e0a-78d3-ad6f-670c3b16c857` |
| 楼名纸条 | V4 · 楼名纸条数据集（新） | `01a1d89f-9084-7a31-b3c0-5ed503fd486b` |
| 禁行牌 | V4 · 禁行牌数据集（新） | `01a1d89f-92ef-7532-8638-fb84176d630d` |
| 垃圾桶状态 | V4 · 垃圾桶状态数据集（新） | `01a1d89f-959c-7400-a557-7e63c927f39c` |
| 垃圾贴纸 | V4 · 垃圾贴纸数据集（新） | `01a1d89f-9813-7c32-8372-ad6843804bf3` |
| 仪表 | V4 · 仪表数据集（新） | `01a1d89f-9a8f-7142-a003-fc67b2af2e79` |

## 窗口通信与共享写入

任务窗口直接对用户交付。跨窗口只传具体问题、相关文件、负责范围和需要协调的写入；共享素材、清单、数据库和同一 Git 索引一次安排一个写者。核实是否为同一 checkout，保留其他窗口的变更。通信本身不会获得文件锁，也不会授予代用户发消息的权限。多个渲染批次可使用独立后台进程共读统一 Blender 工程，按[并行渲染与工程写入](README.md#并行渲染与工程写入)分配输出和保存时段。创建或联系聊天须有用户授权，具体操作见 `workspace-collaboration`。
