# Data, annotation, and asset authorities

Read this reference when reconciling documents with labels, review records, source/split declarations, production receipts, media indexes, or Blender assets. Read `config/workspace_layout.yaml` for the current dataset-task, script, sample, export, and annotation roots; do not assume every task has `datasets/annotated/<task>/`.

## Route to the owning record

| Question | Current authority | What it does not prove |
| --- | --- | --- |
| Directory or task role | `config/workspace_layout.yaml`; generated root README table and `docs/DATASET_DESIGN.md` layout block | That data exists, is labelled, or has been accepted |
| Perception requirement | `config/perception_tasks.json`; generated task READMEs and task matrix sections | Implementation, data, test, or field validation |
| Task inputs, parent groups, split reservation, exclusion | `datasets/scripts/<task>/config/source.yaml`, especially `pairing` | Completed labels or an executed plan; `plan_alignment: stale` remains pending |
| Task plan, budget, and branches | `datasets/scripts/<task>/config/*.json`, as named by the task and its generator | Produced images, label truth, or model quality |
| Batch source and item observations | The batch's existing JSON/JSONL/CSV manifest and its referenced evidence | Human acceptance unless the record names the reviewer and scope |
| Automatic parent-frame review | `datasets/scripts/common/review/`, task `review_plan.json`, tracked task `review_selection.json`, and recomputable observations under `datasets/annotated/common/review/` | Manual acceptance. An empty `accepted_by` is not approval; an automatic `ready` is not complete label truth |
| Human semantic review | The current `manual_review.json`, plus a tracked acceptance record under `datasets/scripts/<task>/config/` when the workflow provides one | Detector box, mask, raw-render, or training eligibility unless the acceptance record explicitly covers that layer |
| Explicit detector labels | Current task annotation JSON/JSONL and label-audit reports; check class order, visible boxes, negative evidence, and review owner | That every unlisted target was absent, or that candidate boxes are human truth |
| Production reuse and prior execution | `datasets/annotated/media_index/production_state.sqlite3` | Semantic correctness or current file freshness; receipts retain historical inputs and outputs |
| Media observation search | `datasets/annotated/media_index/query_cache.sqlite3` | Authority over original manifests or Blender lifecycle. It is a rebuildable snapshot |
| Blender asset lifecycle and reviewed relations | `datasets/blender_source_registry.sqlite3` via `lab/scripts/blender/blender_source_registry.py` | That a binary currently exists; verify files on disk when needed. Original manifests remain generation-time evidence |
| Finished sample or dataset | The current `datasets/samples/<task>/` or `datasets/exports/<task>/` path and its manifest | Label truth, independence, or model performance |

## Annotation state layers

Treat these as independent layers with independent counts and explicit units:

| Layer | What it records | What it does not imply |
| --- | --- | --- |
| `source_available` | A usable source file, frame, record, or candidate input exists at the inspected path | Annotation, validation, confirmation, or training admission |
| `annotated` | The owning annotation schema contains a label for the stated target or a recorded negative | Candidate quality or human confirmation |
| `candidate` | A detector, sampler, or generator proposed an object, region, label, or selection for review | Accepted truth; retain it when later human evidence is absent |
| `auto_validated` | A named programmatic check or rule passed for the recorded scope | Human semantic or geometric confirmation |
| `human_confirmed` | A named reviewer accepted the stated semantic scope, and separately any box, keypoint, mask, or other geometric scope | Any unstated scope or training admission |
| `training_admitted` | The current owning workflow admitted the record to a named training input or split | Model quality, independent capture, or later permission for another purpose |

- `pending` is the unresolved state between or within these layers. An empty `accepted_by`, unreviewed candidate, non-admitted record, or absent evidence of failure remains pending. Use `rejected` or `excluded` only when a record contains a concrete failed check, explicit reviewer decision, or other owning evidence of failure.
- Programmatic checks and templates populate `auto_validated` only. Never use them to populate `human_confirmed`, and never turn missing human work into `rejected`, `excluded`, zero annotations, or an empty candidate set.
- Human semantic confirmation and box/keypoint/mask confirmation are separate facts. Record the reviewer and covered scope; semantic acceptance does not confirm coordinates or masks, and geometric acceptance does not prove the intended object semantics.
- Every annotation or review summary lists all six layer counts and the pending count. Do not abbreviate the report to one “complete labels” total or describe `human_confirmed: 0` as if no annotation records exist.

## Status discipline

- Count files, records, boxes, instances, parent groups, and accepted reviews separately. Keep their units and denominators in the text.
- Distinguish `complete`, `partial`, `unknown`, `confirmed_negative`, candidate-positive, candidate-negative, and excluded records. An empty file is not a negative unless the owning schema and review evidence establish that.
- Keep machine instruction or automatic verdict separate from human confirmation. Review scope matters: semantic object presence may be accepted while coordinates, masks, source renders, or training admission remain pending.
- A missing source, stale index snapshot, failed generator, or unresolved relation stays explicit. Do not convert it to zero, success, or a new source record.
- Summarize only from records actually inspected. State the covered task, batch, path, filesystem or database scope, and exclusions.

## Current command routes

Run from the workspace root:

```powershell
python lab/scripts/contracts/workspace_layout.py check
python lab/scripts/contracts/workspace_ontology.py --check
python lab/scripts/contracts/source_pairing.py --check
python lab/scripts/media/media_index_sqlite.py status
python lab/scripts/blender/blender_source_registry.py list
```

On the Windows checkout, run `state_coverage.py` with `.venv/Scripts/python.exe`; it imports SciPy, and the system interpreter may lack that dependency. A `ModuleNotFoundError` is an environment failure, not evidence that every plan is invalid.

Use the commands and schemas documented in `datasets/scripts/README.md` and `lab/scripts/README.md` for review, label audit, masks, cameras, and sample validation. Do not create a second checker or summarize ignored data by scanning the whole tree.

## Editing boundaries

- Track durable source/split decisions, review plans, and human acceptance in task configuration. Recomputable observations and candidate media may live under ignored `datasets/annotated/` paths.
- Query recorded counts from current owning files or database rows before editing a document. Do not copy an old summary count when the owning records have changed.
- If a summary record omits a current record, state the mismatch and the authority used for the current total; regenerate the summary only through its owning workflow.
- Do not put annotation inventories into the document retriever or a new global registry. Use the batch manifests, source ledger, media index, production receipt database, and Blender registry for their separate responsibilities.
- Review and manual-file reruns are merge-preserve: retain untouched prior entries, candidates, annotations, review decisions, reviewer data, and manual content. A default validation or `--check` path is read-only and must not rewrite those files.
- Rebuilding a review/manual file from a template requires an explicit request. Before replacement, detect existing non-empty manual content and create a recoverable backup; then rebuild only the requested file. If an entry point cannot preserve or back up manual content, do not use it as maintenance and report the implementation gap.
