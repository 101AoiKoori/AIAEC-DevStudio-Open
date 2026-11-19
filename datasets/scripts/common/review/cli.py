"""Command line entry for the cross-task candidate review.

    python -m datasets.scripts.common.review.cli plan        --task <task>
    python -m datasets.scripts.common.review.cli analyze     --task <task> [--overlay]
    python -m datasets.scripts.common.review.cli apply       --task <task>
    python -m datasets.scripts.common.review.cli report      --task <task>
    python -m datasets.scripts.common.review.cli perspective --task <task> [--dry-run]

`plan` reads only the task's review plan and the candidate pool through the
sampling module. `analyze` measures frames and writes metrics/overlays;
`apply` turns metrics into the committed selection; `report` counts stored
verdicts; `perspective` exports reusable ground/horizon templates. No
subcommand writes any other media file.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..images import read_image
from ..paths import WORKSPACE
from .evidence import (
    batch_dir,
    code_hash,
    params_hash,
    read_json,
    read_metrics,
    write_json,
    write_metrics,
)
from .overlays import draw_overlay, write_overlay
from .perspective import build_document as build_perspective_document
from .schema import (
    SCHEMA_VERSION,
    VERDICT_UNKNOWN,
    Box,
    DuplicateReport,
    EmptinessReport,
    FrameRef,
    GroundObservation,
    MetricRecord,
    Verdict,
    to_jsonable,
)

PACKAGE = "datasets.scripts.common.review"
DEFAULT_ROOT = "datasets/annotated/common/review"
PERSPECTIVE_ROOT = "datasets/annotated/common/perspective"


# ---------------------------------------------------------------- paths


def review_plan_path(task: str) -> Path:
    """Return the task's editable review plan location."""

    return WORKSPACE / "datasets" / "scripts" / task / "config" / "review_plan.json"


def default_selection_path(task: str) -> Path:
    """Return the committed selection location for a task."""

    return WORKSPACE / "datasets" / "scripts" / task / "config" / "review_selection.json"


def _workspace_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else WORKSPACE / path


def _plan_path(task: str, plan_path: str | Path | None = None) -> Path:
    return _workspace_path(plan_path) if plan_path else review_plan_path(task)


def _relative(path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return str(path)


def _metrics_dir(plan: Mapping[str, Any], task: str, batch: str) -> Path:
    configured = (plan.get("outputs") or {}).get("metrics_dir")
    if configured:
        return _workspace_path(configured)
    return _workspace_path(batch_dir(task, batch, root=DEFAULT_ROOT))


def _selection_path(plan: Mapping[str, Any], task: str) -> Path:
    configured = (plan.get("outputs") or {}).get("selection")
    if configured:
        return _workspace_path(configured)
    return default_selection_path(task)


def _load_plan(task: str, plan_path: str | Path | None = None) -> dict:
    path = _plan_path(task, plan_path)
    if not path.exists():
        raise FileNotFoundError(f"缺少审查计划: {_relative(path)}")
    plan = read_json(path)
    declared = plan.get("task")
    if declared and declared != task:
        raise ValueError(f"审查计划任务不匹配: {declared!r} != {task!r}")
    return plan


def _batch(plan: Mapping[str, Any], override: str | None) -> str:
    batch = override or plan.get("batch")
    if not batch:
        raise ValueError("审查计划缺少 batch，请用 --batch 指定")
    return str(batch)


def _now() -> str:
    """ISO8601 timestamp with the local timezone offset."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


# ------------------------------------------------------- lazy dependencies


def _require(name: str):
    """Import one review module lazily with an actionable error."""

    module_name = f"{PACKAGE}.{name}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or module_name
        raise RuntimeError(f"缺少审查依赖模块: {name} ({missing})") from exc


def _pick(module: Any, names: Sequence[str]):
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None


def _select_frames(plan: Mapping[str, Any]) -> Sequence[Any]:
    """Return the review frames declared by the plan, else sample the pool.

    An explicit ``frames`` list is the task author's review set and is used
    verbatim so per-frame annotations survive into ``plan.json``. Only a plan
    without frames falls back to sampling the candidate pool.
    """

    declared = plan.get("frames")
    if isinstance(declared, list) and declared:
        frames = []
        for index, record in enumerate(declared):
            if not isinstance(record, Mapping):
                continue
            item = dict(record)
            item.setdefault("record_index", index)
            frames.append(item)
        return frames

    sampling = _require("sampling")
    selector = _pick(sampling, ("select_representative_frames",))
    if selector is None:
        raise RuntimeError("sampling 缺少 select_representative_frames(...) 接口")
    records = _candidate_records(plan)
    frames = [_frame_ref(record) for record in records]
    config = plan.get("sampling") or {}
    if "max_per_parent" not in config:
        raise ValueError("审查计划缺少 sampling.max_per_parent")
    selected = selector(
        frames,
        max_per_parent=int(config.get("max_per_parent", 0)),
        min_gap_seconds=float(config.get("min_gap_seconds", 0.0)),
        seed=int(config.get("seed", 0)),
    )
    max_frames = config.get("max_frames")
    if max_frames is not None:
        selected = list(selected)[: max(0, int(max_frames))]
    return list(selected)


def _candidate_records(plan: Mapping[str, Any]) -> list[dict]:
    value = plan.get("candidate_pool")
    if not value:
        raise ValueError("审查计划缺少 candidate_pool")
    pool = _workspace_path(str(value))
    if not pool.exists():
        raise FileNotFoundError(f"缺少候选池: {_relative(pool)}")
    document = read_json(pool)
    key = str(plan.get("gate_key") or "samples")
    records = document.get(key)
    if isinstance(records, Mapping):
        records = list(records.values())
    if not isinstance(records, list):
        raise ValueError(f"候选池 {_relative(pool)} 缺少列表字段 {key!r}")
    result = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        item = dict(record)
        item.setdefault("record_index", index)
        result.append(item)
    return result


def _measure_functions():
    """Return the geometry and emptiness measurement callables."""

    modules = {}
    missing = []
    for name in ("sampling", "geometry", "emptiness"):
        try:
            modules[name] = _require(name)
        except RuntimeError as exc:
            missing.append(str(exc))
    if missing:
        raise RuntimeError("\n".join(missing))
    geometry = _pick(modules["geometry"], ("estimate_ground_region",))
    emptiness = _pick(modules["emptiness"], ("scan_unlabelled_targets",))
    if geometry is None:
        raise RuntimeError("geometry 缺少 estimate_ground_region(image, *, params=None) 接口")
    if emptiness is None:
        raise RuntimeError("emptiness 缺少 scan_unlabelled_targets(image, *, boxes, templates_bgr, params) 接口")
    return geometry, emptiness


def _emptiness_config(plan: Mapping[str, Any]) -> dict:
    value = plan.get("emptiness")
    return dict(value) if isinstance(value, Mapping) else {}


def _geometry_params(plan: Mapping[str, Any]) -> dict | None:
    value = plan.get("geometry")
    return dict(value) if isinstance(value, Mapping) else None


def _load_templates(plan: Mapping[str, Any]) -> tuple[list, dict]:
    """Load crop templates declared by the plan, with a legacy path fallback.

    ``templates.load_templates`` crops ``[xyxy]`` out of an already-labelled
    frame. Older plans may still list whole template images under
    ``emptiness.templates``; those are only used when no crop resolved. The
    returned summary states the resolved template set for ``manifest.json``.
    """

    module_name = f"{PACKAGE}.templates"
    try:
        templates_mod = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"缺少审查依赖模块: templates ({exc.name or module_name})") from exc
    arrays, info = templates_mod.load_templates(plan, base_dir=WORKSPACE)
    if not arrays:
        for item in _emptiness_config(plan).get("templates") or []:
            value = item.get("path") if isinstance(item, Mapping) else item
            label = item.get("label") if isinstance(item, Mapping) else None
            record = {"label": label, "source": value, "legacy_whole_image": True}
            if not value:
                record.update({"loaded": False, "reason": "incomplete_declaration"})
                info.append(record)
                continue
            try:
                arrays.append(read_image(_workspace_path(str(value))))
            except (OSError, ValueError) as exc:
                record.update({"loaded": False, "reason": f"unreadable: {exc}"})
            else:
                record["loaded"] = True
            info.append(record)
    return arrays, templates_mod.template_summary(info)


def _boxes(items: Any) -> list[Box]:
    boxes = []
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        xyxy = item.get("xyxy") or item.get("box")
        if not isinstance(xyxy, Sequence) or len(xyxy) < 4:
            continue
        boxes.append(
            Box(
                class_name=str(item.get("class_name") or "target"),
                xyxy=tuple(int(round(float(value))) for value in xyxy[:4]),
                physical_instance_id=item.get("physical_instance_id"),
            )
        )
    return boxes


def _plan_boxes(plan: Mapping[str, Any]) -> list[Box]:
    return _boxes(_emptiness_config(plan).get("boxes"))


def _frame_boxes(frame_payload: Mapping[str, Any], fallback: list[Box]) -> list[Box]:
    """Use this frame's own labelled boxes; only fall back to the plan default."""

    own = _boxes(frame_payload.get("existing_targets"))
    return own or fallback


def _measure_geometry(geometry, image, params: dict | None):
    if params is None:
        return geometry(image)
    parameters = inspect.signature(geometry).parameters
    if "params" in parameters:
        return geometry(image, params=params)
    return geometry(image)


def _measure_clearance(geometry, image, patch):
    """Ground-clearance metrics when the geometry module offers them."""

    module = geometry
    if not hasattr(module, "ground_clearance"):
        # The caller may pass the bound estimate function instead of the module.
        try:
            module = _require("geometry")
        except RuntimeError:
            return {}
    try:
        function = _pick(module, ("ground_clearance",))
    except RuntimeError:
        return {}
    try:
        return dict(function(image, patch) or {})
    except (TypeError, ValueError):
        return {}


def _measure_placement(image, patch, config, boxes):
    """Measure placement clutter; return ``None`` when the module is unavailable."""

    if not isinstance(config, Mapping) or not config:
        return None
    try:
        module = _require("placement")
    except RuntimeError:
        return None
    function = getattr(module, "assess_placement", None)
    if not callable(function):
        return None
    result = function(
        image,
        patch,
        target=config.get("target"),
        boxes=boxes,
        params=config.get("params"),
    )
    if result is None:
        return None
    metrics = getattr(result, "metrics", None)
    masks = getattr(result, "masks", None)
    if not isinstance(metrics, Mapping):
        return None
    return {"metrics": dict(metrics), "masks": dict(masks or {})}



def _measure_emptiness(emptiness, image, boxes, templates, params: dict | None, spec: dict | None = None):
    parameters = inspect.signature(emptiness).parameters
    kwargs = {}
    for keyword, value in (("boxes", boxes), ("templates_bgr", templates), ("params", params), ("spec", spec)):
        if keyword in parameters and value is not None:
            kwargs[keyword] = value
    return emptiness(image, **kwargs)


# --------------------------------------------------- record conversion


def _frame_payload(item: Any) -> dict:
    if isinstance(item, FrameRef):
        return to_jsonable(item)
    if isinstance(item, Mapping):
        return dict(item)
    raise TypeError(f"采样结果不是帧记录: {type(item)!r}")


def _frame_ref(payload: Mapping[str, Any]) -> FrameRef:
    if isinstance(payload, FrameRef):
        return payload
    timestamp = payload.get("timestamp_seconds")
    if timestamp is None:
        timestamp = payload.get("source_time_seconds")
    return FrameRef(
        path=str(payload.get("path") or payload.get("image") or ""),
        parent_video=payload.get("parent_video"),
        timestamp_seconds=timestamp,
        record_index=payload.get("record_index"),
        source_group=payload.get("source_group"),
    )


def _ground(value: Any) -> GroundObservation | None:
    if value is None or isinstance(value, GroundObservation):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"ground 不是 GroundObservation: {type(value)!r}")
    return GroundObservation(
        ok=bool(value.get("ok")),
        polygon=[list(point) for point in value.get("polygon") or []],
        foot_xy=value.get("foot_xy"),
        horizon_y=value.get("horizon_y"),
        residual=value.get("residual"),
        reason_codes=list(value.get("reason_codes") or []),
        placement_patch=[list(point) for point in value.get("placement_patch") or []],
        clearance=dict(value.get("clearance") or {}),
    )


def _emptiness(value: Any) -> EmptinessReport | None:
    if value is None or isinstance(value, EmptinessReport):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"emptiness 不是 EmptinessReport: {type(value)!r}")
    return EmptinessReport(
        clean=bool(value.get("clean")),
        suspects=list(value.get("suspects") or []),
        covered_area_fraction=float(value.get("covered_area_fraction") or 0.0),
        reason_codes=list(value.get("reason_codes") or []),
        below_scale=list(value.get("below_scale") or []),
        hard_negatives=list(value.get("hard_negatives") or []),
        target_spec=dict(value.get("target_spec") or {}),
    )


def _duplicates(value: Any) -> DuplicateReport | None:
    if value is None or isinstance(value, DuplicateReport):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"duplicates 不是 DuplicateReport: {type(value)!r}")
    return DuplicateReport(
        group=str(value.get("group") or ""),
        duplicate_of=value.get("duplicate_of"),
        distances={str(key): float(distance) for key, distance in (value.get("distances") or {}).items()},
    )


def _metric_from_payload(payload: Mapping[str, Any]) -> MetricRecord:
    return MetricRecord(
        frame=_frame_ref(payload.get("frame") or {}),
        task=str(payload.get("task") or ""),
        ground=_ground(payload.get("ground")),
        emptiness=_emptiness(payload.get("emptiness")),
        duplicates=_duplicates(payload.get("duplicates")),
        extra=dict(payload.get("extra") or {}),
        code_hash=str(payload.get("code_hash") or ""),
        params=dict(payload.get("params") or {}),
    )


def _verdict_from_payload(payload: Mapping[str, Any]) -> Verdict:
    return Verdict(
        frame=_frame_ref(payload.get("frame") or {}),
        role=str(payload.get("role") or "background"),
        verdict=str(payload.get("verdict") or VERDICT_UNKNOWN),
        reason_codes=list(payload.get("reason_codes") or []),
        ground_polygon=payload.get("ground_polygon"),
        foot_xy=payload.get("foot_xy"),
        existing_targets=list(payload.get("existing_targets") or []),
        near_duplicate_group=payload.get("near_duplicate_group"),
        reviewed_by=str(payload.get("reviewed_by") or "vision"),
        accepted_by=payload.get("accepted_by"),
        evidence=dict(payload.get("evidence") or {}),
    )


def _summarize_verdicts(verdicts: Sequence[Verdict]) -> dict:
    by_verdict: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    by_group: dict[str, int] = {}
    for item in verdicts:
        by_verdict[item.verdict] = by_verdict.get(item.verdict, 0) + 1
        for reason in item.reason_codes:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        group = item.frame.source_group or item.frame.parent_video or "unknown"
        by_group[group] = by_group.get(group, 0) + 1
    return {
        "total": len(verdicts),
        "by_verdict": dict(sorted(by_verdict.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "by_group": dict(sorted(by_group.items())),
    }


def _summarize(module: Any, verdicts: Sequence[Verdict]) -> dict:
    function = _pick(module, ("summarize",))
    return function(verdicts) if function is not None else _summarize_verdicts(verdicts)


def _group_name(frame: Mapping[str, Any]) -> str:
    return str(frame.get("source_group") or frame.get("parent_video") or frame.get("path") or "unknown")


def _code_hash() -> str:
    root = WORKSPACE / "datasets" / "scripts" / "common" / "review"
    paths = sorted(path.relative_to(WORKSPACE).as_posix() for path in root.glob("*.py"))
    return code_hash(paths)


# ------------------------------------------------------------- commands


def _cmd_plan(args: argparse.Namespace) -> int:
    plan = _load_plan(args.task, args.plan)
    batch = _batch(plan, args.batch)
    try:
        selected = _select_frames(plan)
    except RuntimeError as exc:
        print(str(exc))
        return 2
    frames = [_frame_payload(item) for item in selected]
    groups = len({_group_name(frame) for frame in frames})
    print(f"任务: {args.task}")
    print(f"批次: {batch}")
    print(f"父组数: {groups}")
    print(f"帧数: {len(frames)}")
    if args.dry_run:
        print("dry-run: 未写入 plan.json")
        return 0
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": args.task,
        "batch": batch,
        "generated_at": _now(),
        "review_plan": _relative(_plan_path(args.task, args.plan)),
        "candidate_pool": plan.get("candidate_pool"),
        "gate_key": plan.get("gate_key"),
        "sampling": plan.get("sampling"),
        "frames": frames,
    }
    target = _metrics_dir(plan, args.task, batch) / "plan.json"
    write_json(target, payload)
    print(f"已写入: {_relative(target)}")
    return 0


def _overlay_enabled(plan: Mapping[str, Any]) -> bool:
    configuration = plan.get("overlays") or {}
    return configuration.get("enabled", True) is not False


def _overlay_candidate(ground: GroundObservation | None, emptiness: EmptinessReport | None) -> bool:
    """Draw a frame when it could still be ``ready``/``unknown``.

    A usable ground observation keeps the frame in play; without it the judge
    excludes the frame, but reported emptiness suspects are still worth showing
    as visual evidence.
    """

    if ground is not None and bool(ground.ok):
        return True
    return emptiness is not None and bool(emptiness.suspects)


def _overlay_name(index: int, frame_path: str) -> str:
    """ASCII-safe overlay name so cv2.imwrite never sees a non-ANSI stem."""

    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(frame_path).stem or "frame")
    return f"{index:04d}_{stem}.png"


def _cmd_analyze(args: argparse.Namespace) -> int:
    plan = _load_plan(args.task, args.plan)
    batch = _batch(plan, args.batch)
    batch_path = _metrics_dir(plan, args.task, batch)
    plan_file = batch_path / "plan.json"
    if not plan_file.exists():
        print(f"缺少计划文件: {_relative(plan_file)}（先运行 plan）")
        return 1
    plan_document = read_json(plan_file)
    planned = list(plan_document.get("frames") or [])
    frames = planned
    if args.limit is not None:
        frames = frames[: max(0, int(args.limit))]
    try:
        geometry, emptiness = _measure_functions()
    except RuntimeError as exc:
        print(str(exc))
        return 2
    try:
        templates, template_summary = _load_templates(plan)
    except RuntimeError as exc:
        print(str(exc))
        return 2
    except (OSError, ValueError) as exc:
        print(f"错误: 无法读取审查模板: {exc}")
        return 1
    global_boxes = _plan_boxes(plan)
    geometry_params = _geometry_params(plan)
    emptiness_config = _emptiness_config(plan)
    emptiness_params = emptiness_config.get("params")
    emptiness_spec = emptiness_config.get("spec")
    reference_height = emptiness_config.get("reference_frame_height")
    if reference_height is not None and (type(reference_height) is not int or reference_height <= 0):
        print("错误: emptiness.reference_frame_height 必须是正整数")
        return 1
    person_background_params = None
    if args.task == "person" or reference_height is not None:
        from .emptiness import DEFAULT_PARAMS as default_emptiness_params

    if args.task == "person":
        # The two current standee crops are 117–120 px high. The default 0.6
        # scale only examines heights from 70 px; add a bounded distant search.
        person_background_params = dict(emptiness_params or {})
        scales = person_background_params.get("scales", default_emptiness_params["scales"])
        person_background_params["scales"] = tuple(dict.fromkeys((0.2, 0.3, 0.45, *scales)))

    digest = _code_hash()
    parameter_digest = params_hash(plan)
    generated = _now()
    records: list[MetricRecord] = []
    failures: list[dict] = []
    overlay_count = 0
    overlay_dir = batch_path / "overlays"
    for index, item in enumerate(frames):
        frame_path = ""
        try:
            frame_payload = _frame_payload(item)
            frame_path = str(frame_payload.get("path") or "")
            frame = _frame_ref(frame_payload)
            image = read_image(_workspace_path(frame.path))
            ground = _ground(_measure_geometry(geometry, image, geometry_params))
            boxes = _frame_boxes(frame_payload, global_boxes)
            role = str(frame_payload.get("role") or "background")
            person_background = args.task == "person" and role == "background"
            spec = emptiness_spec
            if person_background and reference_height is None:
                base_spec = emptiness_params if spec is None else spec
                # Legacy person plans did not distinguish template-window
                # scale from the scale of an independently measured object.
                spec = {**(base_spec or {}), "below_min_scale": "suspect"}
            frame_params = person_background_params if person_background else emptiness_params
            if reference_height is not None:
                frame_params = dict(frame_params or {})
                scales = frame_params.get("scales", default_emptiness_params["scales"])
                height_ratio = image.shape[0] / reference_height
                frame_params["scales"] = tuple(float(scale) * height_ratio for scale in scales)
            placement_config = plan.get("placement") or {}
            placement_masks = None
            if ground is not None and ground.placement_patch:
                assessed = _measure_placement(
                    image, ground.placement_patch, placement_config, boxes
                )
                if assessed is not None:
                    ground.clearance = assessed["metrics"]
                    placement_masks = assessed["masks"]
                else:
                    ground.clearance = _measure_clearance(
                        geometry, image, ground.placement_patch
                    )
            empty = _emptiness(
                _measure_emptiness(
                    emptiness,
                    image,
                    boxes,
                    templates,
                    frame_params,
                    spec,
                )
            )
            duplicate = _duplicates(frame_payload.get("duplicates")) if "duplicates" in frame_payload else None
            extra = {
                "image_shape": list(image.shape[:2]),
                "parent_link_verified": bool(frame_payload.get("parent_link_verified", True)),
            }
            if args.task == "person" and role != "background":
                extra["role"] = role
            if person_background or reference_height is not None:
                extra["emptiness_search"] = {
                    "method": "template_match",
                    "scales": list(frame_params["scales"]),
                    "template_count": len(templates),
                }
                if reference_height is not None:
                    extra["emptiness_search"]["reference_frame_height"] = reference_height
            if person_background:
                if boxes is not global_boxes:
                    # Only the frame's declarations establish a known target here.
                    extra["existing_targets"] = [asdict(box) for box in boxes]
            record = MetricRecord(
                frame=frame,
                task=args.task,
                ground=ground,
                emptiness=empty,
                duplicates=duplicate,
                extra=extra,
                code_hash=digest,
                params={
                    "sampling": plan.get("sampling"),
                    "thresholds": plan.get("thresholds"),
                    "placement": placement_config,
                },
            )
            records.append(record)
            if args.overlay and _overlay_enabled(plan) and _overlay_candidate(ground, empty):
                overlay = draw_overlay(
                    image,
                    ground=ground,
                    emptiness=empty,
                    duplicates=duplicate,
                    placement_masks=placement_masks,
                    caption=f"{args.task} {Path(frame.path).name}",
                )
                write_overlay(overlay, overlay_dir / _overlay_name(index, frame.path))
                overlay_count += 1
        except Exception as exc:  # one bad frame must not hide the rest
            failures.append({"index": index, "path": frame_path, "error": f"{type(exc).__name__}: {exc}"})

    metrics_path = batch_path / "metrics.jsonl"
    write_metrics(
        metrics_path,
        records,
        header={
            "schema_version": SCHEMA_VERSION,
            "batch": batch,
            "generated_at": generated,
            "frame_count": len(frames),
            "code_hash": digest,
            "params_hash": parameter_digest,
            "review_plan": _relative(_plan_path(args.task, args.plan)),
        },
        task=args.task,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": args.task,
        "batch": batch,
        "generated_at": generated,
        "frame_count": len(planned),
        "processed": len(frames),
        "analyzed": len(records),
        "failed": failures,
        "overlays": overlay_count,
        "templates": template_summary,
        "code_hash": digest,
        "params_hash": parameter_digest,
        "metrics": "metrics.jsonl",
    }
    write_json(batch_path / "manifest.json", manifest)
    print(f"任务: {args.task}")
    print(f"批次: {batch}")
    print(f"帧数: {len(planned)}")
    if args.limit is not None:
        print(f"本次处理: {len(frames)}")
    print(f"已分析: {len(records)}")
    print(f"失败: {len(failures)}")
    if args.overlay:
        print(f"叠加图: {overlay_count}")
    print(f"已写入: {_relative(metrics_path)}")
    return 1 if failures else 0


def _cmd_apply(args: argparse.Namespace) -> int:
    plan = _load_plan(args.task, args.plan)
    batch = _batch(plan, args.batch)
    metrics_path = _metrics_dir(plan, args.task, batch) / "metrics.jsonl"
    if not metrics_path.exists():
        print(f"缺少度量文件: {_relative(metrics_path)}（先运行 analyze）")
        return 1
    header, payloads = read_metrics(metrics_path)
    if not payloads:
        print(f"度量文件没有记录: {_relative(metrics_path)}（未写入 review_selection.json）")
        return 1
    try:
        judge = _require("judge")
    except RuntimeError as exc:
        print(str(exc))
        return 2
    decide = _pick(judge, ("decide",))
    if decide is None:
        print("judge 缺少 decide(metric, plan) 接口")
        return 2
    verdicts = []
    for payload in payloads:
        metric = _metric_from_payload(payload)
        role = metric.extra.get("role", "background") if metric.task == "person" else "background"
        verdict = decide(metric, plan, role=role)
        if metric.emptiness is not None:
            verdict.evidence["automatic_empty_observation"] = to_jsonable(metric.emptiness)
            if metric.code_hash:
                verdict.evidence["automatic_observation_code_hash"] = metric.code_hash
        if metric.ground and metric.ground.ok and metric.ground.placement_patch and metric.ground.foot_xy:
            verdict.evidence["measured_ground_polygon"] = metric.ground.polygon
            verdict.evidence["measured_placement_patch"] = metric.ground.placement_patch
            verdict.evidence["measured_ground_foot_xy"] = metric.ground.foot_xy
        verdicts.append(verdict)
    stats = _summarize(judge, verdicts)
    if args.dry_run:
        print(f"任务: {args.task}")
        print(f"批次: {batch}")
        _print_stats(stats)
        print("dry-run: 未写入 review_selection.json")
        return 0
    selection = {
        "schema_version": SCHEMA_VERSION,
        "task": args.task,
        "batch": batch,
        "code_hash": (header or {}).get("code_hash") or _code_hash(),
        "generated_at": _now(),
        "records": [to_jsonable(verdict) for verdict in verdicts],
    }
    target = _selection_path(plan, args.task)
    if target.exists():
        previous = read_json(target)
        if previous.get("batch") == batch:
            prior_records = previous.get("records") or []
            prior_by_path = {record.get("frame", {}).get("path"): record for record in prior_records}
            current_paths = {record.get("frame", {}).get("path") for record in selection["records"]}
            for record in selection["records"]:
                prior = prior_by_path.get(record.get("frame", {}).get("path"))
                if not prior:
                    continue
                prior_evidence = prior.get("evidence") or {}
                current_evidence = record["evidence"]
                for key in ("source_record", "source_batch"):
                    if key in prior_evidence:
                        current_evidence.setdefault(key, prior_evidence[key])
                history = list(prior_evidence.get("previous_acceptances") or [])
                if prior.get("accepted_by"):
                    scope = prior_evidence.get("acceptance_scope")
                    if prior.get("verdict") == record.get("verdict"):
                        record["accepted_by"] = prior["accepted_by"]
                        if scope is not None:
                            current_evidence["acceptance_scope"] = scope
                    else:
                        old_acceptance = {"verdict": prior.get("verdict"), "accepted_by": prior["accepted_by"]}
                        if scope is not None:
                            old_acceptance["acceptance_scope"] = scope
                        if old_acceptance not in history:
                            history.append(old_acceptance)
                if history:
                    current_evidence["previous_acceptances"] = history
                if prior_evidence.get("parent_relation_verified_by"):
                    current_evidence["parent_relation_verified_by"] = prior_evidence["parent_relation_verified_by"]
            selection["records"].extend(record for record in prior_records
                                        if record.get("frame", {}).get("path") not in current_paths)
            if "previous_batches" in previous:
                selection["previous_batches"] = previous["previous_batches"]
        else:
            selection["previous_batches"] = [*(previous.get("previous_batches") or []),
                                              {key: previous[key] for key in ("batch", "records") if key in previous}]
    write_json(target, selection)
    print(f"任务: {args.task}")
    print(f"批次: {batch}")
    _print_stats(stats)
    print(f"已写入: {_relative(target)}")
    return 0


def _print_stats(stats: Mapping[str, Any]) -> None:
    print(f"判定计数: {stats.get('total', 0)}")
    for key, value in (stats.get("by_verdict") or {}).items():
        print(f"  {key}: {value}")
    print("原因码计数:")
    for key, value in (stats.get("by_reason") or {}).items():
        print(f"  {key}: {value}")
    print("父组计数:")
    for key, value in (stats.get("by_group") or {}).items():
        print(f"  {key}: {value}")


def _cmd_report(args: argparse.Namespace) -> int:
    plan: dict | None = None
    try:
        plan = _load_plan(args.task, args.plan)
    except (OSError, ValueError):
        plan = None
    selection = _selection_path(plan, args.task) if plan else default_selection_path(args.task)
    if not selection.exists():
        print(f"未找到审查判定: {_relative(selection)}（先运行 apply --task {args.task}）")
        return 0
    payload = read_json(selection)
    records = payload.get("records") or []
    try:
        judge = _require("judge")
    except RuntimeError as exc:
        print(str(exc))
        return 2
    verdicts = [_verdict_from_payload(record) for record in records]
    stats = _summarize(judge, verdicts)
    print(f"任务: {args.task}")
    print(f"批次: {payload.get('batch')}")
    _print_stats(stats)
    if plan is not None:
        batch = str(plan.get("batch") or "")
        configured = (plan.get("outputs") or {}).get("metrics_dir")
        if batch or configured:
            metrics_path = _metrics_dir(plan, args.task, batch) / "metrics.jsonl"
            if metrics_path.exists():
                _, rows = read_metrics(metrics_path)
                print(f"已分析帧: {len(rows)}")
            else:
                print(f"度量文件: 缺失（{_relative(metrics_path)}）")
    return 0


def _cmd_perspective(args: argparse.Namespace) -> int:
    plan = _load_plan(args.task, args.plan)
    batch = _batch(plan, args.batch)
    metrics_path = _metrics_dir(plan, args.task, batch) / "metrics.jsonl"
    if not metrics_path.exists():
        print(f"缺少度量文件: {_relative(metrics_path)}（先运行 analyze）")
        return 1
    header, payloads = read_metrics(metrics_path)
    document = build_perspective_document(header or {}, payloads)
    document["source_metrics"] = _relative(metrics_path)
    summary = document["summary"]
    target = _workspace_path(args.output or f"{PERSPECTIVE_ROOT}/{args.task}/{batch}.json")
    print(f"任务: {args.task}")
    print(f"批次: {batch}")
    print(f"测量帧: {summary['measured']}  有地面: {summary['with_ground']}  无地面: {summary['without_ground']}")
    if args.dry_run:
        print(f"dry-run: 未写入 {_relative(target)}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, document)
    print(f"已写入: {_relative(target)}")
    return 0


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m {PACKAGE}.cli", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="select candidate frames into plan.json")
    plan.add_argument("--task", required=True)
    plan.add_argument("--batch")
    plan.add_argument("--plan", help="alternate review plan JSON; defaults to the task plan")
    plan.add_argument("--dry-run", action="store_true")
    plan.set_defaults(handler=_cmd_plan)

    analyze = subparsers.add_parser("analyze", help="measure planned frames")
    analyze.add_argument("--task", required=True)
    analyze.add_argument("--batch")
    analyze.add_argument("--plan", help="alternate review plan JSON; defaults to the task plan")
    analyze.add_argument("--limit", type=int)
    analyze.add_argument("--overlay", action="store_true")
    analyze.set_defaults(handler=_cmd_analyze)

    apply = subparsers.add_parser("apply", help="turn metrics into review_selection.json")
    apply.add_argument("--task", required=True)
    apply.add_argument("--batch")
    apply.add_argument("--plan", help="alternate review plan JSON; defaults to the task plan")
    apply.add_argument("--dry-run", action="store_true")
    apply.set_defaults(handler=_cmd_apply)

    report = subparsers.add_parser("report", help="count stored verdicts")
    report.add_argument("--task", required=True)
    report.add_argument("--plan", help="alternate review plan JSON; defaults to the task plan")
    report.set_defaults(handler=_cmd_report)

    perspective = subparsers.add_parser(
        "perspective", help="export reusable ground/horizon templates from metrics")
    perspective.add_argument("--task", required=True)
    perspective.add_argument("--batch")
    perspective.add_argument("--plan", help="alternate review plan JSON; defaults to the task plan")
    perspective.add_argument("--output")
    perspective.add_argument("--dry-run", action="store_true")
    perspective.set_defaults(handler=_cmd_perspective)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"错误: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
