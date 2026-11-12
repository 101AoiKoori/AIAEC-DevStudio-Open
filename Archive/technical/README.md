# 技术迭代归档

> 本文提及的部分本地素材、数据和历史运行产物未随本仓库提供；相关名称保留用于说明来源和实验背景。

本目录面向国赛工作空间中发生的数据集、脚本、模型等技术迭代，供国赛技术写作对照；文档不参与归档。操作规范见 [历史归档 Skill](../../.agents/skills/workspace-history-archive/SKILL.md)。

- [版本入口](revisions/README.md)：按技术对象与来源定位可复原版本。
- [迭代对照](iterations/README.md)：按主题对照版本变化与验证证据。
- [脚本版本](script-versions.tsv)及[相邻差异](script-differences.tsv)：Git 中多次变化的识别实验、ROS 包及工作空间维护 Python 文件。
- [技术配置版本](config-versions.tsv)：定位各版本的原始来源及本机文件。
- [数据集版本](dataset-versions.tsv)：国赛工作空间产生的数据集版本、来源和状态。
- [素材入口](materials/README.md)：统一查找原始素材并核对来源。
- `assets/`：本机保存的技术模型和对应实验元数据，按任务和对象分类，来源对应关系见索引；不进入 Git。
- `datasets/`：本工作空间产生的技术历史数据，按任务及版本分类；数据不进入 Git。


## 历史数据集 ZIP

history_datasets_20261112.zip 保存本工作空间的历史数据集，大小 16,089,956,910 字节，约 14.98 GiB。归档含 143,966 项，涵盖 15 组已归档数据、18 个 V1/V2 运行的数据子树、已有明确历史状态的候选和国赛 V3 输入快照。已逐项核对路径、解压大小和全量 CRC。

成员使用工作空间相对路径。包内 `_archive_metadata/history_dataset_scope.json` 记录来源、路径、状态与归档范围。现有数据及其引用沿用原路径。原始采集、独立省赛及 CCPD 整包、`training_packages/` 分发镜像、模型、权重、脚本、文档和可重建缓存在各自原目录管理。

Git 保存已追踪脚本的版本；数据和模型版本依据本机文件及实验记录核对。
