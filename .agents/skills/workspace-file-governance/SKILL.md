---
name: workspace-file-governance
description: "Choose and check workspace file locations before experiments, scripting, implementation, imports, moves, or placement audits; preserve ownership, references, and runtime boundaries."
---

# Workspace file governance

Use this skill at the start of experiments, scripting, implementation, and other tasks likely to create, import, move, or reorganize workspace files, even when the user does not mention file management. Read the root `AGENTS.md` and directory table and rules in `README.md` as the workspace authority; use `workspace-doc-governance` for related documentation and requirement updates.

## Decide placement

Identify each item's purpose, origin, owner, lifecycle, deployment role, sensitivity, and whether it may be committed. Choose the existing directory matching those facts before writing. Distinguish deployable package code, experiments, project documentation, external references, original materials, datasets, model weights, and temporary outputs. Do not infer that an external resource is redistributable; record known origin, version and license, and mark unknowns. Do not add runtime dependencies on lab, references or datasets. Create a directory only for a distinct role and update its owning directory description.

Do not reference local paths outside this workspace in scripts or non-root documentation, or load or execute external script files. When external source code is needed, first establish origin and permission, copy the necessary code to its appropriate workspace location and maintain an independent adaptation. Resolve user-supplied script paths and symlinks before loading. Installed interpreters, programs and libraries are not external script files; source attribution links and workspace-relative references remain valid.

Before creating a file, infer its role from the user's task and call `python .agents/skills/workspace-file-governance/scripts/check_layout.py --suggest <kind> [--task <id>]` for an applicable role. Kinds are `training-script`, `training-config`, `training-test`, `inference-script`, `inference-config`, `inference-test` (each requires an ID in `config/perception_tasks.json`), `shared-core`, `package-test`, `navigation`, `integration`, `repo-tool`, and `temporary`. Shared inference belongs in the existing `aec_vision` core even when developed during an experiment; the experiment's driver and evaluation stay in `lab/`. For another ROS package, external materials, data or a new task, use `README.md` and the relevant owner rather than inventing an ID or treating a generic suggestion as authoritative. If intent cannot be inferred, report the missing information; do not guess.

Check the resulting filename with `python .agents/skills/workspace-file-governance/scripts/check_layout.py --path <relative-path>`. An OK result only means no encoded rule matched. A local-only output is a review warning rather than a forbidden destination, but must not be committed. Run without `--path` for an audit, and with `--staged` to check staged changes and remaining files in affected packages. Exit 0 means no mechanically detected violations, 1 means violations, and 2 means the check could not complete. Review warnings and explain unresolved cases. The local Git hook runs the staged check if `core.hooksPath` is configured as `.githooks`; hooks can be disabled or bypassed and are not a security boundary.

The checker inspects path names, not file contents or semantic correctness. Its default scan covers tracked and non-ignored untracked files plus top-level directories; it does not traverse ignored data, secrets, model or temporary trees, and does not detect all runtime dependencies, ignored files or misplaced but plausible files. Staged symlinks and submodules are flagged for review; their contents and targets are not inspected by the staged check. It never moves or deletes anything. An agent must still inspect task-specific ownership and references.

## Archive and lifecycle ownership

For obsolete authored plans, preserve still-valid facts in their current owner and use Git for prior document history. Update active inbound links and generator inputs; do not leave active duplicate advice. Technical iteration archives exclude documents. Raw inputs, current experiment evidence and user assets retain their own lifecycle. Runtime images/video belong to the configured run output area; memory release, temporary-file disposal and retention of required original evidence are separate operations.

Resolve ordinary placement from existing directory ownership and proceed. A checker error must be resolved before placing or staging the affected file; a review warning needs a purpose/ownership check and can be resolved from existing evidence. A new path alone does not require permission. An OK result still requires checking content and runtime dependencies relevant to the task.

## Change and inspect

- Inspect Git status first and preserve unrelated or uncommitted work. Search actual consumers, paths, imports, launch/config references, scripts, generators, tests and docs before moving anything. Consider relative path resolution, platform case sensitivity and deploy packaging.
- Move only items whose ownership and destination are established and whose relocation is within the user's task. Repair references and relevant documentation together. Preserve external originals and user data; do not move or delete unknown, sensitive or ignored files just to make the tree tidy.
- After placement changes, the owning directory description should state current locations, purposes and any pending cleanup targets or dependencies. Use links from other documents rather than repeating inventories or move/delete histories. Keep an unfinished cleanup recorded until its outcome is verified; a documentation edit does not establish that files were moved or removed. Apply the document governance editing criteria.
- For generated or temporary output, use existing ignored locations. Check ignore rules and staged files; ignoring a path does not untrack already tracked files or guarantee secrets are safe. Do not commit data, weights, credentials or build outputs.
- Review the changed paths and references, run checks appropriate to the affected operation, and follow the repository's focused Git commit rule. When a source, destination, ownership or dependency is uncertain, leave the affected item in place and report the obstacle, impact and evidence or decision needed. A scan is scoped: state exclusions and do not claim the whole tree is clean without evidence.
