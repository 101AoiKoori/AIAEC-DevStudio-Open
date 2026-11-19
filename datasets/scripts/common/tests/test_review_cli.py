"""Contract tests for the review command line entry point."""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from datasets.scripts.common.review import cli, evidence
from datasets.scripts.common.review.schema import EmptinessReport, FrameRef, GroundObservation, MetricRecord, Verdict


def write_review_plan(tmp, task="person", batch="unit_batch"):
    path = Path(tmp) / "review_plan.json"
    evidence.write_json(
        path,
        {
            "schema_version": 1,
            "task": task,
            "batch": batch,
            "candidate_pool": "datasets/annotated/person/candidates.json",
            "gate_key": "samples",
            "sampling": {"max_per_parent": 2, "min_gap_seconds": 10.0, "seed": 1918, "max_frames": 3},
            "thresholds": {"ground_min_area_ratio": 0.02},
            "overlays": {"enabled": True, "verdicts": ["ready", "unknown"]},
        },
    )
    return path


def frames():
    return [
        {"path": "frames/a.png", "source_group": "g1"},
        {"path": "frames/b.png", "source_group": "g1"},
        {"path": "frames/c.png", "source_group": "g2"},
    ]


class SelectionIntegrationTest(unittest.TestCase):
    def test_explicit_plan_frames_are_used_verbatim(self):
        plan = {
            "task": "person",
            "frames": [
                {
                    "path": "frames/a.png",
                    "source_group": "g1",
                    "parent_link_verified": False,
                    "role": "background",
                    "existing_targets": [{"class_name": "community", "xyxy": [1, 2, 3, 4]}],
                }
            ],
            "sampling": {"max_per_parent": 0},
        }
        selected = cli._select_frames(plan)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["existing_targets"],
                         [{"class_name": "community", "xyxy": [1, 2, 3, 4]}])
        self.assertFalse(selected[0]["parent_link_verified"])
        self.assertEqual(selected[0]["role"], "background")

    def test_per_frame_boxes_prefer_own_targets_then_global(self):
        from datasets.scripts.common.review.schema import Box

        global_boxes = [Box(class_name="fallback", xyxy=(0, 0, 10, 10))]
        own = cli._frame_boxes(
            {"existing_targets": [{"class_name": "community", "xyxy": [1, 2, 3, 4]}]},
            global_boxes,
        )
        self.assertEqual([box.class_name for box in own], ["community"])
        self.assertEqual(own[0].xyxy, (1, 2, 3, 4))
        self.assertIs(cli._frame_boxes({"existing_targets": []}, global_boxes), global_boxes)

    def test_candidate_pool_is_sampled_through_the_sampling_module(self):
        with TemporaryDirectory() as tmp:
            pool = Path(tmp) / "candidates.json"
            records = [
                {"image": "frames/g1_000.png", "source_group": "g1", "source_time_seconds": 0.0},
                {"image": "frames/g1_003.png", "source_group": "g1", "source_time_seconds": 3.0},
                {"image": "frames/g1_010.png", "source_group": "g1", "source_time_seconds": 10.0},
                {"image": "frames/g1_020.png", "source_group": "g1", "source_time_seconds": 20.0},
                {"image": "frames/g2_000.png", "source_group": "g2", "source_time_seconds": 0.0},
            ]
            evidence.write_json(pool, {"version": 1, "samples": records})
            plan = {
                "task": "person",
                "candidate_pool": str(pool),
                "gate_key": "samples",
                "sampling": {
                    "max_per_parent": 2,
                    "min_gap_seconds": 5.0,
                    "seed": 1918,
                    "max_frames": 2,
                },
            }
            selected = cli._select_frames(plan)
            self.assertLessEqual(len(selected), 2)
            self.assertTrue(all(frame.source_group == "g1" for frame in selected))
            self.assertTrue(all(frame.path.startswith("frames/g1_") for frame in selected))
            self.assertTrue(all(frame.record_index is not None for frame in selected))

    def test_load_templates_crops_declared_box(self):
        import cv2
        import numpy as np

        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "labelled.png"
            image = np.zeros((40, 50, 3), np.uint8)
            image[10:30, 5:25] = (0, 0, 255)
            cv2.imwrite(str(source), image)
            plan = {"templates": [{"label": "target", "source": str(source), "xyxy": [5, 10, 25, 30]}]}
            arrays, summary = cli._load_templates(plan)
            self.assertEqual(len(arrays), 1)
            self.assertEqual(arrays[0].shape, (20, 20, 3))
            self.assertEqual(summary["loaded"], 1)
            self.assertEqual(summary["labels"], ["target"])


class AnalyzeRecordTest(unittest.TestCase):
    def test_analyze_uses_frame_boxes_and_parent_link_flag(self):
        import cv2
        import numpy as np

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            image_path = tmp / "frame.png"
            cv2.imwrite(str(image_path), np.zeros((60, 80, 3), np.uint8))
            config = write_review_plan(tmp)
            plan = evidence.read_json(config)
            plan["emptiness"] = {
                "reference_frame_height": 30,
                "spec": {"min_height_fraction": 0.28, "below_min_scale": "suspect"},
            }
            evidence.write_json(config, plan)
            batch = tmp / "batch"
            batch.mkdir()
            frame = {
                "path": str(image_path),
                "source_group": "g1",
                "parent_link_verified": False,
                "existing_targets": [{"class_name": "community", "xyxy": [1, 2, 3, 4]}],
            }
            evidence.write_json(batch / "plan.json", {"task": "person", "frames": [frame]})
            seen = {}

            def fake_geometry(image, params=None):
                return GroundObservation(ok=True, polygon=[[1, 1], [70, 1], [70, 50], [1, 50]],
                                         foot_xy=[30, 40])

            def fake_emptiness(image, *, boxes, templates_bgr, params=None, spec=None):
                seen["boxes"] = list(boxes)
                seen["params"] = params
                seen["spec"] = spec
                return EmptinessReport(clean=False, suspects=[{"xyxy": [2, 2, 8, 8], "score": 0.9}])

            output = io.StringIO()
            with mock.patch.object(cli, "review_plan_path", return_value=config), \
                    mock.patch.object(cli, "batch_dir", return_value=batch), \
                    mock.patch.object(cli, "_measure_functions", return_value=(fake_geometry, fake_emptiness)), \
                    mock.patch.object(cli, "_load_templates", return_value=([], {"declared": 0, "loaded": 0, "labels": [], "failures": []})), \
                    mock.patch.object(cli, "_code_hash", return_value="0123456789abcdef"), \
                    redirect_stdout(output):
                code = cli.main(["analyze", "--task", "person", "--overlay"])
            self.assertEqual(code, 0)
            self.assertEqual([box.class_name for box in seen["boxes"]], ["community"])
            self.assertEqual(seen["spec"], {"min_height_fraction": 0.28, "below_min_scale": "suspect"})
            self.assertEqual(seen["params"]["scales"][:3], (0.4, 0.6, 0.9))
            header, rows = evidence.read_metrics(batch / "metrics.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["extra"]["parent_link_verified"], False)
            self.assertEqual(rows[0]["extra"]["image_shape"], [60, 80])
            self.assertEqual(rows[0]["extra"]["existing_targets"], [{
                "class_name": "community", "xyxy": [1, 2, 3, 4], "physical_instance_id": None,
            }])
            self.assertEqual(rows[0]["extra"]["emptiness_search"]["scales"], list(seen["params"]["scales"]))
            self.assertEqual(rows[0]["extra"]["emptiness_search"]["reference_frame_height"], 30)
            manifest = evidence.read_json(batch / "manifest.json")
            self.assertEqual(manifest["templates"]["loaded"], 0)
            self.assertEqual(len(list((batch / "overlays").glob("*.png"))), 1)


class PlanCommandTest(unittest.TestCase):
    def test_alternate_plan_path_overrides_the_task_default(self):
        with TemporaryDirectory() as tmp:
            config = write_review_plan(tmp, batch="alternate_batch")
            batch = Path(tmp) / "alternate_batch"
            output = io.StringIO()
            with mock.patch.object(cli, "review_plan_path", side_effect=AssertionError("default plan accessed")), \
                    mock.patch.object(cli, "_select_frames", return_value=frames()), \
                    mock.patch.object(cli, "batch_dir", return_value=batch), \
                    redirect_stdout(output):
                code = cli.main(["plan", "--task", "person", "--plan", str(config)])
            self.assertEqual(code, 0)
            self.assertEqual(evidence.read_json(batch / "plan.json")["batch"], "alternate_batch")
            self.assertIn("批次: alternate_batch", output.getvalue())

    def test_dry_run_returns_zero_and_writes_nothing(self):
        with TemporaryDirectory() as tmp:
            config = write_review_plan(tmp)
            batch = Path(tmp) / "batch"
            output = io.StringIO()
            with mock.patch.object(cli, "review_plan_path", return_value=config), \
                    mock.patch.object(cli, "_select_frames", return_value=frames()), \
                    mock.patch.object(cli, "batch_dir", return_value=batch), \
                    redirect_stdout(output):
                code = cli.main(["plan", "--task", "person", "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(batch.exists())
            text = output.getvalue()
            self.assertIn("任务: person", text)
            self.assertIn("批次: unit_batch", text)
            self.assertIn("父组数: 2", text)
            self.assertIn("帧数: 3", text)
            self.assertIn("dry-run", text)

    def test_plan_writes_frames_into_batch_dir(self):
        with TemporaryDirectory() as tmp:
            config = write_review_plan(tmp)
            batch = Path(tmp) / "batch"
            with mock.patch.object(cli, "review_plan_path", return_value=config), \
                    mock.patch.object(cli, "_select_frames", return_value=frames()), \
                    mock.patch.object(cli, "batch_dir", return_value=batch), \
                    redirect_stdout(io.StringIO()):
                code = cli.main(["plan", "--task", "person"])
            self.assertEqual(code, 0)
            plan_file = batch / "plan.json"
            self.assertTrue(plan_file.exists())
            payload = evidence.read_json(plan_file)
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["task"], "person")
            self.assertEqual(payload["batch"], "unit_batch")
            self.assertEqual([frame["path"] for frame in payload["frames"]],
                             ["frames/a.png", "frames/b.png", "frames/c.png"])

    def test_batch_override_is_used(self):
        with TemporaryDirectory() as tmp:
            config = write_review_plan(tmp)
            batch = Path(tmp) / "other_batch"
            output = io.StringIO()
            with mock.patch.object(cli, "review_plan_path", return_value=config), \
                    mock.patch.object(cli, "_select_frames", return_value=frames()), \
                    mock.patch.object(cli, "batch_dir", return_value=batch), \
                    redirect_stdout(output):
                code = cli.main(["plan", "--task", "person", "--batch", "override", "--dry-run"])
            self.assertEqual(code, 0)
            self.assertIn("批次: override", output.getvalue())


class AnalyzeDependencyTest(unittest.TestCase):
    def test_missing_modules_return_two_and_write_nothing(self):
        with TemporaryDirectory() as tmp:
            config = write_review_plan(tmp)
            batch = Path(tmp) / "batch"
            batch.mkdir()
            evidence.write_json(batch / "plan.json", {"task": "person", "frames": frames()})
            output = io.StringIO()

            def missing(name):
                raise RuntimeError(f"缺少审查依赖模块: {name} (not installed)")

            with mock.patch.object(cli, "review_plan_path", return_value=config), \
                    mock.patch.object(cli, "batch_dir", return_value=batch), \
                    mock.patch.object(cli, "_require", side_effect=missing), \
                    redirect_stdout(output):
                code = cli.main(["analyze", "--task", "person"])
            self.assertEqual(code, 2)
            text = output.getvalue()
            for name in ("sampling", "geometry", "emptiness"):
                self.assertIn(name, text)
            self.assertFalse((batch / "metrics.jsonl").exists())
            self.assertFalse((batch / "manifest.json").exists())
            self.assertFalse((batch / "overlays").exists())


class ReportCommandTest(unittest.TestCase):
    def test_missing_selection_prints_hint_and_returns_zero(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "missing_review_plan.json"
            selection = Path(tmp) / "review_selection.json"
            output = io.StringIO()
            with mock.patch.object(cli, "review_plan_path", return_value=config), \
                    mock.patch.object(cli, "default_selection_path", return_value=selection), \
                    redirect_stdout(output):
                code = cli.main(["report", "--task", "person"])
            self.assertEqual(code, 0)
            text = output.getvalue()
            self.assertIn("未找到审查判定", text)
            self.assertIn("review_selection.json", text)
            self.assertFalse(selection.exists())

    def test_report_counts_stored_verdicts(self):
        with TemporaryDirectory() as tmp:
            selection = Path(tmp) / "review_selection.json"
            evidence.write_json(
                selection,
                {
                    "schema_version": 1,
                    "task": "person",
                    "batch": "unit_batch",
                    "code_hash": "abcd",
                    "records": [
                        {
                            "frame": {"path": "frames/a.png", "source_group": "g1"},
                            "role": "background",
                            "verdict": "ready",
                            "reason_codes": ["no_unlabelled_target"],
                        },
                        {
                            "frame": {"path": "frames/b.png", "source_group": "g2"},
                            "role": "background",
                            "verdict": "unknown",
                            "reason_codes": ["emptiness_unproven"],
                        },
                    ],
                },
            )
            fake_judge = SimpleNamespace(summarize=lambda verdicts: _fake_summary(verdicts))
            output = io.StringIO()
            with mock.patch.object(cli, "review_plan_path", return_value=Path(tmp) / "none.json"), \
                    mock.patch.object(cli, "default_selection_path", return_value=selection), \
                    mock.patch.object(cli, "_require", return_value=fake_judge), \
                    redirect_stdout(output):
                code = cli.main(["report", "--task", "person"])
            self.assertEqual(code, 0)
            text = output.getvalue()
            self.assertIn("判定计数: 2", text)
            self.assertIn("ready: 1", text)
            self.assertIn("unknown: 1", text)
            self.assertIn("no_unlabelled_target: 1", text)
            self.assertIn("g1: 1", text)


class ApplyCommandTest(unittest.TestCase):
    def test_new_machine_observation_updates_background_with_uncertain_placement(self):
        import cv2
        import numpy as np

        from datasets.scripts.person import compile_background_dispatch as dispatch

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_review_plan(root)
            plan = evidence.read_json(config)
            plan["placement"] = {"required": True}
            evidence.write_json(config, plan)
            batch = root / "batch"
            selection = root / "review_selection.json"
            task_plan = root / "production_plan.json"
            source = root / "source.yaml"
            output = root / "background_dispatch.json"
            image = np.zeros((60, 80, 3), np.uint8)
            polygon = [[5, 20], [75, 20], [75, 57], [5, 57]]
            patch = [[12, 28], [68, 28], [78, 55], [2, 55]]
            existing_box = {"class_name": "community", "xyxy": [4, 15, 12, 45]}
            paths = ("frame.png", "known_label.png", "new_suspect.png", "old_measurement.png")
            source_rows = []
            measures = []
            for path in paths:
                self.assertTrue(cv2.imwrite(str(root / path), image))
                source_rows.append({
                    "image": path, "split": "val", "source_group": "held_out",
                    "state": "candidate_complete_negative" if path == "new_suspect.png" else "unknown",
                    "semantic_status": "unknown",
                    "boxes": [existing_box] if path == "known_label.png" else [],
                })
                suspects = [{"xyxy": [4, 15, 12, 45], "score": 0.85}] if path == "new_suspect.png" else []
                extra = {"image_shape": [60, 80]}
                if path != "old_measurement.png":
                    extra["emptiness_search"] = {"reference_frame_height": 360}
                measures.append(MetricRecord(
                    frame=FrameRef(path=path, source_group="observed"),
                    task="person",
                    ground=GroundObservation(ok=True, polygon=polygon, placement_patch=patch, foot_xy=[39, 47]),
                    emptiness=EmptinessReport(clean=not suspects, suspects=suspects),
                    extra=extra, code_hash="new_measurement",
                ))
            evidence.write_metrics(batch / "metrics.jsonl", measures, task="person")
            evidence.write_json(selection, {"batch": "unit_batch", "records": [{
                "frame": {"path": paths[0]}, "verdict": "unknown",
                "evidence": {"automatic_empty_observation": {"clean": False}},
            }]})
            evidence.write_json(task_plan, {"model_key": "person", "target_split": "train", "strata": [{}]})
            evidence.write_json(output, {"split_assignment": "source_pairing", "records": source_rows})
            source.write_text("roles:\n  reviewed_background: existing.json\n", encoding="utf-8")

            with mock.patch.object(cli, "review_plan_path", return_value=config), \
                    mock.patch.object(cli, "batch_dir", return_value=batch), \
                    mock.patch.object(cli, "default_selection_path", return_value=selection), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["apply", "--task", "person"]), 0)
            reviewed = evidence.read_json(selection)["records"][0]
            self.assertEqual(reviewed["reason_codes"], ["placement_patch_unsure"])
            self.assertTrue(reviewed["evidence"]["automatic_empty_observation"]["clean"])
            self.assertEqual(reviewed["evidence"]["measured_placement_patch"], patch)

            with mock.patch.multiple(dispatch, ROOT=root, SELECTION=selection, PLAN=task_plan,
                                     SOURCE=source, DISPATCH=output):
                dispatch.compile_dispatch(update_plan=False)
            updated, untouched, contradicted, old_only = evidence.read_json(output)["records"]
            self.assertEqual(updated["state"], "candidate_complete_negative")
            self.assertEqual(updated["semantic_status"], "auto_candidate_complete_negative")
            self.assertEqual(updated["review_verdict"], "unknown")
            self.assertEqual(updated["automatic_observation_code_hash"], "new_measurement")
            self.assertIsNone(updated["accepted_by"])
            self.assertEqual((updated["split"], updated["source_group"]), ("val", "held_out"))
            self.assertEqual(untouched, source_rows[1])
            self.assertEqual(contradicted["state"], "unknown")
            self.assertEqual(contradicted["review_reason_codes"], ["unlabelled_target_suspected"])
            self.assertEqual(old_only, source_rows[3])

    def test_apply_preserves_same_batch_acceptance(self):
        with TemporaryDirectory() as tmp:
            config = write_review_plan(tmp)
            batch = Path(tmp) / "batch"
            batch.mkdir()
            record = MetricRecord(frame=FrameRef(path="frames/a.png", source_group="g1"), task="person")
            evidence.write_metrics(batch / "metrics.jsonl", [record],
                                   header={"batch": "unit_batch", "code_hash": "abcd"}, task="person")
            selection = Path(tmp) / "review_selection.json"
            evidence.write_json(selection, {"batch": "unit_batch", "records": [{
                "frame": {"path": "frames/a.png"}, "verdict": "ready",
                "accepted_by": "reviewer", "evidence": {"acceptance_scope": "verdict"}}]})
            fake_judge = SimpleNamespace(
                decide=lambda metric, plan=None, role="background": Verdict(
                    frame=metric.frame, role=role, verdict="ready"),
                summarize=lambda verdicts: _fake_summary(verdicts),
            )
            with mock.patch.object(cli, "review_plan_path", return_value=config), \
                    mock.patch.object(cli, "batch_dir", return_value=batch), \
                    mock.patch.object(cli, "default_selection_path", return_value=selection), \
                    mock.patch.object(cli, "_require", return_value=fake_judge), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["apply", "--task", "person"]), 0)
            actual = evidence.read_json(selection)["records"][0]
            self.assertEqual(actual["accepted_by"], "reviewer")
            self.assertEqual(actual["evidence"]["acceptance_scope"], "verdict")

    def test_apply_keeps_changed_verdict_acceptance_as_history(self):
        with TemporaryDirectory() as tmp:
            config = write_review_plan(tmp)
            batch = Path(tmp) / "batch"
            record = MetricRecord(frame=FrameRef(path="frames/a.png"), task="person")
            evidence.write_metrics(batch / "metrics.jsonl", [record], task="person")
            selection = Path(tmp) / "review_selection.json"
            untouched = {"frame": {"path": "frames/b.png"}, "verdict": "ready", "accepted_by": "reviewer"}
            evidence.write_json(selection, {"batch": "unit_batch", "records": [{
                "frame": {"path": "frames/a.png"}, "verdict": "ready", "accepted_by": "reviewer",
                "evidence": {"acceptance_scope": "placement", "parent_relation_verified_by": "owner"},
            }, untouched], "previous_batches": [{"batch": "earlier"}]})
            with mock.patch.object(cli, "review_plan_path", return_value=config), \
                    mock.patch.object(cli, "batch_dir", return_value=batch), \
                    mock.patch.object(cli, "default_selection_path", return_value=selection), \
                    redirect_stdout(io.StringIO()):
                for _ in range(2):
                    self.assertEqual(cli.main(["apply", "--task", "person"]), 0)
            result = evidence.read_json(selection)
            changed, remaining = result["records"]
            self.assertEqual(changed["verdict"], "unknown")
            self.assertIsNone(changed["accepted_by"])
            self.assertNotIn("acceptance_scope", changed["evidence"])
            self.assertEqual(changed["evidence"]["previous_acceptances"], [{
                "verdict": "ready", "accepted_by": "reviewer", "acceptance_scope": "placement",
            }])
            self.assertEqual(changed["evidence"]["parent_relation_verified_by"], "owner")
            self.assertEqual(remaining, untouched)
            self.assertEqual(result["previous_batches"], [{"batch": "earlier"}])

    def test_dry_run_uses_judge_without_writing_selection(self):
        with TemporaryDirectory() as tmp:
            config = write_review_plan(tmp)
            batch = Path(tmp) / "batch"
            batch.mkdir()
            record = MetricRecord(
                frame=FrameRef(path="frames/a.png", source_group="g1"),
                task="person",
                extra={"image_shape": [120, 160]},
                code_hash="abcd",
            )
            evidence.write_metrics(batch / "metrics.jsonl", [record],
                                   header={"batch": "unit_batch", "code_hash": "abcd"}, task="person")
            selection = Path(tmp) / "review_selection.json"

            def decide(metric, plan=None, role="background"):
                return Verdict(frame=metric.frame, role=role, verdict="ready",
                               reason_codes=["no_unlabelled_target"])

            fake_judge = SimpleNamespace(decide=decide, summarize=lambda verdicts: _fake_summary(verdicts))
            output = io.StringIO()
            with mock.patch.object(cli, "review_plan_path", return_value=config), \
                    mock.patch.object(cli, "batch_dir", return_value=batch), \
                    mock.patch.object(cli, "default_selection_path", return_value=selection), \
                    mock.patch.object(cli, "_require", return_value=fake_judge), \
                    redirect_stdout(output):
                code = cli.main(["apply", "--task", "person", "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(selection.exists())
            text = output.getvalue()
            self.assertIn("判定计数: 1", text)
            self.assertIn("dry-run", text)


def _fake_summary(verdicts):
    by_verdict = {}
    by_reason = {}
    by_group = {}
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


if __name__ == "__main__":
    unittest.main()
