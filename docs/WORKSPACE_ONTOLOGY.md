# 工作空间对象分类学

> 自动生成：修改 `config/workspace_ontology.yaml` 后运行 `python lab/scripts/contracts/workspace_ontology.py --write`。本文件不是目录契约；目录结构见 `config/workspace_layout.yaml`。

工作空间对象类层次与路径归属；目录结构由 config/workspace_layout.yaml 管，本文管对象语义。

## 类层次

| 类 | 父类 | 抽象 | 所有者前缀 | 权威源 | 版本策略 | 可变性 | Git | 生命周期 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| artifact | - | 是 | - | none | none | abstract | mixed | - |
| source_artifact | artifact | 是 | - | manifest | manifest | immutable | mixed | captured, catalogued, retired |
| raw_media | source_artifact | 否 | datasets/raw/, Archive/Assets/captures/ | manifest | manifest | immutable | False | captured, catalogued, retired |
| capture_batch | source_artifact | 否 | datasets/CAR_REAL_CAPTURE_, datasets/SITE_CAPTURE_ | manifest | manifest | immutable | True | captured, catalogued, retired |
| blender_asset | source_artifact | 否 | datasets/assets/blender/ | blender_registry | registry | versioned | False | captured, catalogued, retired |
| candidate_source_asset | source_artifact | 否 | datasets/assets/person/, Archive/Assets/working_sources/ | manifest | manifest | immutable | False | catalogued, retired |
| derived_artifact | artifact | 是 | - | manifest | manifest | rebuildable | False | candidate, reviewed, accepted, superseded |
| candidate_batch | derived_artifact | 否 | datasets/annotated/ | manifest | manifest | rebuildable | False | candidate, reviewed, accepted, isolated, superseded |
| review_observation | derived_artifact | 否 | datasets/annotated/common/review/, datasets/annotated/common/perspective/ | manifest | manifest | rebuildable | False | generated, reviewed, superseded |
| source_ledger | derived_artifact | 否 | datasets/annotated/common/sources/ | manifest | manifest | rebuildable | False | generated, superseded |
| record_artifact | artifact | 是 | - | mixed | mixed | append_only | mixed | active, superseded |
| media_index_store | record_artifact | 否 | datasets/annotated/media_index/ | media_index | manifest | mixed | False | active, superseded |
| media_index_cache | media_index_store | 否 | - | media_index | manifest | rebuildable | False | active, superseded |
| production_receipt | media_index_store | 否 | - | production_receipt | receipt | append_only | False | active, superseded |
| frame_environment | record_artifact | 否 | - | mixed | frame_environment | versioned | False | active, superseded |
| asset_registry | record_artifact | 否 | - | blender_registry | registry | append_only | True | active, superseded |
| review_decision | record_artifact | 否 | - | git | git | versioned | True | current, superseded |
| plan_artifact | artifact | 是 | - | git | git | versioned | True | draft, reviewed, approved, active, executed, superseded |
| requirement_config | plan_artifact | 否 | - | git | git | versioned | True | active, superseded |
| layout_contract | plan_artifact | 否 | - | git | git | versioned | True | active, superseded |
| ontology_contract | plan_artifact | 否 | - | git | git | versioned | True | active, superseded |
| design_contract | plan_artifact | 否 | - | git | git | versioned | True | active, superseded |
| design_observation_snapshot | plan_artifact | 否 | - | git | git | versioned | True | active, superseded |
| task_config | plan_artifact | 否 | - | git | git | versioned | True | active, superseded |
| plan_draft | plan_artifact | 否 | - | git | git | versioned | True | design_draft, reviewed, approved, compiled, superseded |
| generation_plan | plan_artifact | 否 | - | git | git | versioned | True | active, executed, superseded |
| production_plan | generation_plan | 否 | - | git | git | versioned | True | active, executed, superseded |
| review_plan | generation_plan | 否 | - | git | git | versioned | True | active, executed, superseded |
| dataset_artifact | artifact | 是 | - | manifest | manifest | current_only | False | work_in_progress, review_ready, usable, isolated, superseded |
| sample_set | dataset_artifact | 否 | datasets/samples/ | manifest | manifest | current_only | False | work_in_progress, review_ready, usable, isolated |
| dataset_export | dataset_artifact | 否 | datasets/exports/ | manifest | manifest | current_only | False | work_in_progress, review_ready, usable, isolated, superseded |
| training_package | dataset_artifact | 否 | training_packages/ | manifest | manifest | rebuildable | False | active, superseded |
| model_artifact | artifact | 是 | - | model_record | model | versioned | False | trained, evaluated, deployed, retired |
| training_run | model_artifact | 否 | runs/ | run_record | run | immutable | False | running, completed, failed, interrupted |
| model_weight | model_artifact | 否 | models/weights/ | model_record | model | versioned | False | trained, evaluated, retired |
| deployed_model | model_artifact | 否 | ws/src/aec_vision/models/ | model_record | model | current_only | False | deployed, retired |
| code_artifact | artifact | 是 | - | git | git | versioned | True | active, superseded |
| deploy_package | code_artifact | 否 | ws/src/ | git | git | versioned | True | active, superseded |
| experiment_code | code_artifact | 否 | lab/features/, lab/navigation/, lab/integration/, lab/training_common/ | git | git | versioned | True | active, superseded |
| data_pipeline_code | code_artifact | 否 | datasets/scripts/ | git | git | versioned | True | active, superseded |
| repo_tool | code_artifact | 否 | lab/scripts/ | git | git | versioned | True | active, superseded |
| skill_script | code_artifact | 否 | .agents/skills/ | git | git | versioned | True | active, superseded |
| doc_artifact | artifact | 是 | - | git | git | versioned | True | current, archived |
| project_doc | doc_artifact | 否 | docs/ | git | git | versioned | True | current, archived |
| generated_doc | doc_artifact | 否 | - | git | git | generated | True | current, superseded |
| reference_doc | doc_artifact | 否 | refs/ | git | git | versioned | True | current, archived |
| archive_artifact | artifact | 是 | - | archive_index | archive | immutable | mixed | archived |
| technical_archive | archive_artifact | 否 | Archive/technical/ | archive_index | archive | immutable | mixed | archived |
| external_original | archive_artifact | 否 | Archive/Assets/ | archive_index | archive | immutable | mixed | archived |
| publication_export | artifact | 否 | AIAEC-DevStudio-Open/ | external_repo | external | versioned | False | active, superseded |
| temp_artifact | artifact | 否 | tmp/ | none | none | disposable | False | disposable |
| app_runtime | artifact | 否 | .claude/ | none | none | disposable | False | transient |

## 关系

| 关系 | 从 | 到 | 说明 |
| --- | --- | --- | --- |
| derived_from | candidate_batch, review_observation, source_ledger, sample_set, dataset_export, training_run, model_weight, production_receipt | raw_media, capture_batch, blender_asset, candidate_source_asset, candidate_batch, sample_set, dataset_export, generation_plan | 派生对象回指父来源、父批次或生成计划。 |
| annotates | candidate_batch | raw_media, capture_batch, blender_asset, candidate_source_asset | 标注/候选批次标注其来源素材。 |
| reviews | review_observation, review_decision, review_plan | candidate_batch | 审查观测、判定和计划作用于候选批次。 |
| plans | generation_plan, production_plan, review_plan, task_config, plan_draft | requirement_config, design_contract, task_config | 计划/任务配置引用需求与设计契约。 |
| produces | generation_plan, production_plan, data_pipeline_code, production_receipt | sample_set, dataset_export | 生成计划或数据管线产出当前小样/正式集合。 |
| trains | training_run, experiment_code | dataset_export | 训练运行/实验代码消费正式集合。 |
| deploys | deployed_model, deploy_package | model_weight | 部署模型来自研究/训练权重。 |
| documents | project_doc, generated_doc, reference_doc, technical_archive | artifact | 文档记录或引用任意对象。 |
| supersedes | artifact | artifact | 新版本取代旧版本；历史由 Git 或归档承载。 |

## 版本策略

| 策略 | 说明 |
| --- | --- |
| git | 文本、代码、配置和文档由 Git 保存当前版本；不建版本命名目录，历史用提交。 |
| manifest | 数据/标注用清单记录来源、派生、用途和状态；当前集合只保留一套。 |
| registry | Blender 资产状态以 blender_source_registry.sqlite3 为权威。 |
| frame_environment | 逐图数据库维护当前任务用途资格，资格与证据修订 ID 供选源、生成和历史回执引用。 |
| receipt | production_state.sqlite3 保存持久生产回执，不当可重建缓存清理。 |
| run | runs 内运行记录保持原位；内层 V3 runs 不移动、删除或覆盖。 |
| model | 模型权重按来源、训练数据版本、类别顺序和运行记录管理；大文件不入 Git。 |
| archive | 退役技术迭代进入 Archive/technical，保留原路径和来源关系。 |
| mixed | 目录内不同文件有不同权威/版本策略，详见类说明。 |
| external | 由外部独立仓库或发布流程管理，不在本仓库内版本化。 |
| none | 不保留版本，可随时丢弃。 |

## 关键对象映射

| 路径或模式 | 类 |
| --- | --- |
| `config/workspace_ontology.yaml` | ontology_contract |
| `config/workspace_layout.yaml` | layout_contract |
| `config/perception_tasks.json` | requirement_config |
| `config/task_targets.yaml` | design_contract |
| `config/id_policy.yaml` | design_contract |
| `config/combination_axes.yaml` | design_contract |
| `config/combination_axis_observations.yaml` | design_observation_snapshot |
| `config/full_train_allocation.yaml` | plan_draft |
| `docs/FULL_TRAIN_PLAN_DESIGN.md` | generated_doc |
| `docs/ID_AND_COUNT_POLICY.md` | generated_doc |
| `docs/WORKSPACE_ONTOLOGY.md` | generated_doc |
| `docs/TASK_MATRIX.md` | generated_doc |
| `lab/features/README.md` | generated_doc |
| `datasets/blender_source_registry.sqlite3` | asset_registry |
| `datasets/annotated/media_index/query_cache.sqlite3` | media_index_cache |
| `datasets/annotated/media_index/production_state.sqlite3` | production_receipt |
| `datasets/annotated/common/sources/frame_environment.sqlite3` | frame_environment |
| `datasets/CAR_REAL_CAPTURE_20261112.md` | capture_batch |
| `datasets/SITE_CAPTURE_20261112.md` | capture_batch |
| `datasets/README.md` | project_doc |
| `datasets/scripts/README.md` | project_doc |
| `models/README.md` | project_doc |
| `README.md` | project_doc |
| `AGENTS.md` | project_doc |
| `datasets/DATASET_COORDINATION*.md` | project_doc |
| `.agents/skills/*/SKILL.md` | project_doc |
| `datasets/scripts/*/config/production_plan.json` | production_plan |
| `datasets/scripts/*/config/review_plan.json` | review_plan |
| `datasets/scripts/*/config/review_selection.json` | review_decision |
| `datasets/scripts/*/config/dataset.yaml` | task_config |
| `datasets/scripts/*/config/source.yaml` | task_config |
| `datasets/scripts/*/config/*.json` | generation_plan |

## 检查规则

- 必需对象：`config/workspace_layout.yaml`、`config/workspace_ontology.yaml`、`config/task_targets.yaml`、`config/combination_axes.yaml`、`config/full_train_allocation.yaml`、`config/id_policy.yaml`、`docs/FULL_TRAIN_PLAN_DESIGN.md`、`docs/ID_AND_COUNT_POLICY.md`、`docs/WORKSPACE_ONTOLOGY.md`
- 生成文档：`docs/WORKSPACE_ONTOLOGY.md`、`docs/FULL_TRAIN_PLAN_DESIGN.md`、`docs/ID_AND_COUNT_POLICY.md`
- 计划草案状态：`config/full_train_allocation.yaml` 的 `status` 必须为 `design_draft`。

## 新增对象

1. 在 `classes` 中选择父类，定义 `owner_prefixes`、`authority`、`versioning`、`mutability`、`tracked`、`lifecycle` 和 `relations`。
2. 在 `artifacts` 中添加精确路径或模式；同一路径只能属于一个类。
3. 运行 `python lab/scripts/contracts/workspace_ontology.py --write` 生成本文件，再运行 `--check`。
4. 不另建与现有类同义的目录、清单或版本命名目录；历史由 Git 或归档承载。
