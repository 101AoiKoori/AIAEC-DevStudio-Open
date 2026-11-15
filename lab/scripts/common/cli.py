"""Command line entry for workspace-wide dataset authoring aids.

    python -m lab.scripts.common.cli background discover ...
    python -m lab.scripts.common.cli background summarize ...

The CLI writes evidence and reports only. It does not create training samples,
assign human acceptance, launch production, or train a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .background_pool import (
    discover_media_candidates,
    summarize_review,
    write_background_report,
)
from .blender_cameras import validate_camera_plan
from .branch_sample import (
    OBJECT_MATRIX_COLUMNS,
    backup_manual_review,
    label_manifest_from_review,
    load_review_answers,
    merge_review_preserving_existing,
    normalize_review_answers,
    rebuild_review_template,
    review_acceptance_document,
    review_status_layers,
    validate_manual_review,
    validate_sample,
    write_review_template_if_missing,
)
from .contracts import ROOT, atomic_write_json, read_json, resolve_workspace, workspace_relative
from .labels import audit_manifest as audit_labels
from .labels import rebuild_labels
from .mask_quality import audit_manifest as audit_masks

PACKAGE = "lab.scripts.common.cli"


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _output(value: str | None, *, task: str, batch: str, name: str) -> str:
    return value or f"datasets/annotated/common/review/{task}/{batch}/{name}"


def _background_discover(args: argparse.Namespace) -> int:
    report = discover_media_candidates(
        args.database,
        contains=args.contains,
        media_type=args.media_type,
        limit=args.limit,
        root=ROOT,
    )
    target = _output(args.output, task=args.task, batch=args.batch, name="background_candidates.json")
    path = atomic_write_json(target, report, root=ROOT)
    _emit(
        {
            "status": "written",
            "output": workspace_relative(path, root=ROOT),
            "candidates": report["summary"]["returned"],
            "semantic_state": report["summary"]["semantic_state"],
        }
    )
    return 0


def _background_summarize(args: argparse.Namespace) -> int:
    report = summarize_review(
        task=args.task,
        batch=args.batch,
        review_plan=args.review_plan,
        review_selection=args.review_selection,
        metrics=args.metrics,
        root=ROOT,
    )
    target = _output(args.output, task=args.task, batch=args.batch, name="background_pool.json")
    path = write_background_report(report, target, root=ROOT)
    _emit(
        {
            "status": "written",
            "output": workspace_relative(path, root=ROOT),
            "summary": report["summary"],
        }
    )
    return 0


def _label_audit(args: argparse.Namespace) -> int:
    manifest = read_json(args.manifest, root=ROOT)
    report = audit_labels(manifest, root=ROOT)
    target = _output(
        args.output,
        task=str(manifest.get("task") or "common"),
        batch=str(manifest.get("batch") or "label_review"),
        name="label_audit.json",
    )
    path = atomic_write_json(target, report, root=ROOT)
    rebuilt = []
    if args.rebuild_labels:
        rebuilt = rebuild_labels(report, args.rebuild_labels, root=ROOT)
    _emit(
        {
            "status": "written",
            "output": workspace_relative(path, root=ROOT),
            "summary": report["summary"],
            "rebuilt_labels": rebuilt,
        }
    )
    return 1 if report["summary"]["by_status"].get("reject") else 0


def _mask_audit(args: argparse.Namespace) -> int:
    manifest = read_json(args.manifest, root=ROOT)
    task = str(manifest.get("task") or "common")
    batch = str(manifest.get("batch") or "mask_review")
    report_target = _output(args.output, task=task, batch=batch, name="mask_quality.json")
    report_path = resolve_workspace(report_target, root=ROOT)
    overlay_dir = None
    if args.overlay:
        overlay_dir = str(resolve_workspace(args.overlay, root=ROOT))
    report = audit_masks(manifest, root=ROOT, overlay_dir=overlay_dir)
    path = atomic_write_json(report_target, report, root=ROOT)
    _emit(
        {
            "status": "written",
            "output": workspace_relative(path, root=ROOT),
            "overlay_dir": workspace_relative(overlay_dir, root=ROOT) if overlay_dir else None,
            "summary": report["summary"],
        }
    )
    return 1 if report["summary"].get("reject") else 0


def _camera_validate(args: argparse.Namespace) -> int:
    plan = read_json(args.plan, root=ROOT)
    report = validate_camera_plan(plan, root=ROOT)
    target = _output(
        args.output,
        task=str(plan.get("task") or "common"),
        batch=str(plan.get("id") or "camera_plan"),
        name="camera_validation.json",
    )
    path = atomic_write_json(target, report, root=ROOT)
    _emit(
        {
            "status": report["status"],
            "output": workspace_relative(path, root=ROOT),
            "errors": report["errors"],
            "warnings": report["warnings"],
            "coverage": report["coverage"],
        }
    )
    return 1 if report["status"] == "fail" else 0


def _manual_review_payload(report: dict, *, task: str, batch: str) -> dict:
    return {
        "schema_version": report["schema_version"],
        "generated_at": report["generated_at"],
        "task": task,
        "batch": batch,
        "instructions": (
            "逐张查看原图和全帧框叠图，在 status_matrix 中填写 presence、"
            "task_target_count、background_state、accepted_by、review_state 和 "
            "full_frame_checked。objects 每行对应一个独立物体；坐标写入 "
            "visible_bbox_xyxy，有完整实体依据时再写 full_bbox_xyxy。"
            "accepted_by 只确认语义，box_accepted_by 只确认框；未确认保持 pending。"
        ),
        "object_matrix_columns": OBJECT_MATRIX_COLUMNS,
        "items": report["manual_review"],
    }


def _sample_validate(args: argparse.Namespace) -> int:
    manifest = read_json(args.manifest, root=ROOT)
    report = validate_sample(manifest, root=ROOT)
    task = str(manifest.get("task") or "common")
    batch = str(manifest.get("batch") or "branch_sample")
    target = _output(args.output, task=task, batch=batch, name="sample_validation.json")
    path = atomic_write_json(target, report, root=ROOT)
    review_path = path.with_name("manual_review.json")
    review_payload = _manual_review_payload(report, task=task, batch=batch)
    review_created = write_review_template_if_missing(review_path, review_payload, root=ROOT)
    review_backup = None
    if not review_created:
        existing = read_json(review_path, root=ROOT)
        merged = merge_review_preserving_existing(existing, review_payload)
        if merged != existing:
            backup = backup_manual_review(review_path, root=ROOT)
            atomic_write_json(review_path, merged, root=ROOT)
            review_backup = workspace_relative(backup, root=ROOT) if backup else None
    _emit(
        {
            "status": report["status"],
            "output": workspace_relative(path, root=ROOT),
            "manual_review": workspace_relative(review_path, root=ROOT),
            "manual_review_template_created": review_created,
            "manual_review_backup": review_backup,
            "errors": report["errors"],
            "warnings": report["warnings"],
            "summary": report["summary"],
            "status_layers": report["status_layers"],
        }
    )
    return 1 if report["status"] == "fail" else 0


def _sample_rebuild_review_template(args: argparse.Namespace) -> int:
    manifest = read_json(args.manifest, root=ROOT)
    report = validate_sample(manifest, root=ROOT)
    task = str(manifest.get("task") or "common")
    batch = str(manifest.get("batch") or "branch_sample")
    target = args.output or _output(
        None,
        task=task,
        batch=batch,
        name="manual_review.json",
    )
    payload = _manual_review_payload(report, task=task, batch=batch)
    result = rebuild_review_template(target, payload, root=ROOT)
    _emit({"status": "written", **result})
    return 0


def _sample_normalize_review(args: argparse.Namespace) -> int:
    review = read_json(args.review, root=ROOT)
    answers = load_review_answers(args.answers, root=ROOT)
    normalized = normalize_review_answers(review, answers)
    validation = validate_manual_review(normalized)
    target = args.output or str(
        Path(args.review).with_name("manual_review.normalized.json")
    )
    path = atomic_write_json(target, normalized, root=ROOT)
    _emit(
        {
            "status": validation["status"],
            "output": workspace_relative(path, root=ROOT),
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "status_layers": validation["status_layers"],
        }
    )
    return 1 if validation["status"] == "fail" else 0


def _sample_export_label_manifest(args: argparse.Namespace) -> int:
    review = read_json(args.review, root=ROOT)
    if args.class_names:
        review["class_names"] = [
            item.strip() for item in args.class_names.split(",") if item.strip()
        ]
    manifest = label_manifest_from_review(review)
    target = args.output or str(
        Path(args.review).with_name("label_manifest.from_review.json")
    )
    path = atomic_write_json(target, manifest, root=ROOT)
    _emit(
        {
            "status": "written",
            "output": workspace_relative(path, root=ROOT),
            "records": len(manifest["records"]),
            "states": dict(
                sorted(
                    (
                        state,
                        sum(record["label_state"] == state for record in manifest["records"]),
                    )
                    for state in {
                        record["label_state"] for record in manifest["records"]
                    }
                )
            ),
            "skipped": manifest["skipped"],
            "status_layers": review_status_layers(review),
        }
    )
    return 0


def _sample_check_review(args: argparse.Namespace) -> int:
    review = read_json(args.review, root=ROOT)
    report = validate_manual_review(review)
    target = args.output or str(Path(args.review).with_name("manual_review_check.json"))
    path = atomic_write_json(target, report, root=ROOT)
    _emit(
        {
            "status": report["status"],
            "output": workspace_relative(path, root=ROOT),
            "errors": report["errors"],
            "warnings": report["warnings"],
            "items": report["items"],
            "status_layers": report["status_layers"],
        }
    )
    return 1 if report["status"] == "fail" else 0


def _sample_accept_review(args: argparse.Namespace) -> int:
    review = read_json(args.review, root=ROOT)
    validation = validate_manual_review(review)
    accepted = review_acceptance_document(
        review,
        source_review=args.review,
        validation=validation,
        root=ROOT,
    )
    path = atomic_write_json(args.output, accepted, root=ROOT)
    _emit(
        {
            "status": accepted["status"],
            "output": workspace_relative(path, root=ROOT),
            "source_review": accepted["source_review"],
            "source_sha256": accepted["source_sha256"],
            "items": len(accepted["items"]),
            "status_layers": accepted["status_layers"],
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m {PACKAGE}", description=__doc__)
    groups = parser.add_subparsers(dest="command", required=True)
    background = groups.add_parser("background", help="background pool discovery and review")
    actions = background.add_subparsers(dest="action", required=True)

    discover = actions.add_parser("discover", help="query image candidates from the media cache")
    discover.add_argument("--database", default="datasets/annotated/media_index/query_cache.sqlite3")
    discover.add_argument("--contains")
    discover.add_argument("--media-type", default="image")
    discover.add_argument("--limit", type=int, default=200)
    discover.add_argument("--task", default="common")
    discover.add_argument("--batch", default="background_discovery")
    discover.add_argument("--output")
    discover.set_defaults(handler=_background_discover)

    summarize = actions.add_parser("summarize", help="merge review verdicts with source pairing")
    summarize.add_argument("--task", required=True)
    summarize.add_argument("--batch", required=True)
    summarize.add_argument("--review-plan", required=True)
    summarize.add_argument("--review-selection", required=True)
    summarize.add_argument("--metrics")
    summarize.add_argument("--output")
    summarize.set_defaults(handler=_background_summarize)

    label = groups.add_parser("labels", help="audit complete, partial and negative labels")
    label_actions = label.add_subparsers(dest="action", required=True)
    label_audit = label_actions.add_parser("audit", help="validate a label manifest and optionally rebuild YOLO labels")
    label_audit.add_argument("--manifest", required=True)
    label_audit.add_argument("--output")
    label_audit.add_argument("--rebuild-labels")
    label_audit.set_defaults(handler=_label_audit)

    mask = groups.add_parser("masks", help="audit full, visible and occlusion mask layers")
    mask_actions = mask.add_subparsers(dest="action", required=True)
    mask_audit = mask_actions.add_parser("audit", help="measure mask geometry and write an evidence report")
    mask_audit.add_argument("--manifest", required=True)
    mask_audit.add_argument("--output")
    mask_audit.add_argument("--overlay")
    mask_audit.set_defaults(handler=_mask_audit)

    camera = groups.add_parser("cameras", help="validate Blender master-scene camera plans")
    camera_actions = camera.add_subparsers(dest="action", required=True)
    camera_validate = camera_actions.add_parser("validate", help="check projection, views, scale and registration")
    camera_validate.add_argument("--plan", required=True)
    camera_validate.add_argument("--output")
    camera_validate.set_defaults(handler=_camera_validate)

    sample = groups.add_parser("sample", help="validate a bounded branch-complete sample")
    sample_actions = sample.add_subparsers(dest="action", required=True)
    sample_validate = sample_actions.add_parser("validate", help="check source/content branches and write a manual review queue")
    sample_validate.add_argument("--manifest", required=True)
    sample_validate.add_argument("--output")
    sample_validate.set_defaults(handler=_sample_validate)

    sample_rebuild = sample_actions.add_parser(
        "rebuild-review-template",
        help="explicitly rebuild a manual review template, backing up edited content",
    )
    sample_rebuild.add_argument("--manifest", required=True)
    sample_rebuild.add_argument("--output")
    sample_rebuild.set_defaults(handler=_sample_rebuild_review_template)

    sample_normalize = sample_actions.add_parser(
        "normalize-review",
        help="apply JSON or CSV decisions without replacing the source review",
    )
    sample_normalize.add_argument("--review", required=True)
    sample_normalize.add_argument("--answers", required=True)
    sample_normalize.add_argument("--output")
    sample_normalize.set_defaults(handler=_sample_normalize_review)

    sample_export = sample_actions.add_parser(
        "export-label-manifest",
        help="translate normalized single-frame review rows into an explicit label manifest",
    )
    sample_export.add_argument("--review", required=True)
    sample_export.add_argument("--output")
    sample_export.add_argument(
        "--class-names",
        help="comma-separated class names, required when the review file omits class_names",
    )
    sample_export.set_defaults(handler=_sample_export_label_manifest)

    sample_check = sample_actions.add_parser("check-review", help="validate a filled manual review JSON")
    sample_check.add_argument("--review", required=True)
    sample_check.add_argument("--output")
    sample_check.set_defaults(handler=_sample_check_review)

    sample_accept = sample_actions.add_parser("accept-review", help="persist a passing manual review as tracked decisions")
    sample_accept.add_argument("--review", required=True)
    sample_accept.add_argument(
        "--output",
        default="datasets/scripts/person/config/toolchain_review_acceptance.json",
    )
    sample_accept.set_defaults(handler=_sample_accept_review)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
