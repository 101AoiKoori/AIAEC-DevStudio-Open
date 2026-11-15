"""Contract tests for review evidence files (JSONL, hashes, JSON documents)."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets.scripts.common.review import evidence
from datasets.scripts.common.review.schema import (
    EmptinessReport,
    FrameRef,
    GroundObservation,
    MetricRecord,
    to_jsonable,
)


def make_record(path="frames/a.png", note="人物", score=0.8):
    return MetricRecord(
        frame=FrameRef(
            path=path,
            parent_video="clip.mp4",
            timestamp_seconds=1.5,
            record_index=3,
            source_group="group_1",
        ),
        task="person",
        ground=GroundObservation(
            ok=True,
            polygon=[[10, 20], [30, 20], [30, 60], [10, 60]],
            foot_xy=[20, 55],
            residual=0.4,
        ),
        emptiness=EmptinessReport(
            clean=False,
            suspects=[{"xyxy": [12, 22, 28, 58], "score": score}],
            covered_area_fraction=0.12,
        ),
        extra={"image_shape": [120, 160], "note": note},
        code_hash="0123456789abcdef",
        params={"seed": 1918},
    )


class MetricsRoundTripTest(unittest.TestCase):
    def test_write_and_read_with_header(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "metrics.jsonl"
            records = [make_record("a.png"), make_record("b.png", note="第二帧")]
            written = evidence.write_metrics(
                target,
                records,
                header={"batch": "b1", "code_hash": "abcd"},
                task="person",
            )
            self.assertEqual(written, 2)
            self.assertTrue(target.exists())

            raw = target.read_bytes()
            self.assertNotIn(b"\r", raw)
            self.assertTrue(raw.endswith(b"\n"))
            text = raw.decode("utf-8")
            self.assertIn("人物", text)
            self.assertNotIn("\\u4eba", text)

            lines = text.splitlines()
            self.assertEqual(len(lines), 3)
            first = _json_line(lines[0])
            self.assertEqual(first["kind"], "header")
            self.assertEqual(first["task"], "person")
            self.assertEqual(first["batch"], "b1")
            self.assertEqual(first["code_hash"], "abcd")
            self.assertEqual(list(_json_line(lines[1])), sorted(_json_line(lines[1])))

            header, loaded = evidence.read_metrics(target)
            self.assertEqual(header["kind"], "header")
            self.assertEqual(header["batch"], "b1")
            self.assertEqual(loaded, [to_jsonable(record) for record in records])

    def test_header_is_optional_and_blank_lines_are_skipped(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "metrics.jsonl"
            self.assertEqual(evidence.write_metrics(target, [make_record()]), 1)
            header, loaded = evidence.read_metrics(target)
            self.assertIsNone(header)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["frame"]["path"], "frames/a.png")

            with target.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write("\n\n")
            header, loaded = evidence.read_metrics(target)
            self.assertIsNone(header)
            self.assertEqual(len(loaded), 1)

    def test_bad_line_reports_line_number(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "metrics.jsonl"
            target.write_text('{"kind": "header"}\n\n{broken json\n', encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                evidence.read_metrics(target)
            self.assertIn("line 3", str(caught.exception))

            target.write_text('{"kind": "header"}\n[1, 2]\n', encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                evidence.read_metrics(target)
            self.assertIn("line 2", str(caught.exception))

    def test_metric_records_without_header_keep_kind_out(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "metrics.jsonl"
            evidence.write_metrics(target, [make_record()], task="")
            line = target.read_text(encoding="utf-8").strip()
            self.assertNotIn('"kind"', line)


class HashTest(unittest.TestCase):
    def test_params_hash_ignores_key_order(self):
        left = {"b": [1, 2], "a": {"y": 2, "x": 1}, "中文": "值"}
        right = {"a": {"x": 1, "y": 2}, "中文": "值", "b": [1, 2]}
        self.assertEqual(evidence.params_hash(left), evidence.params_hash(right))
        self.assertNotEqual(evidence.params_hash({"seed": 1}), evidence.params_hash({"seed": 2}))
        self.assertEqual(len(evidence.params_hash({})), 16)
        int(evidence.params_hash(left), 16)

    def test_code_hash_is_stable_and_content_sensitive(self):
        with TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.py"
            second = Path(tmp) / "second.py"
            first.write_text("VALUE = 1\n", encoding="utf-8")
            second.write_text("OTHER = 2\n", encoding="utf-8")

            baseline = evidence.code_hash([first, second])
            self.assertEqual(len(baseline), 16)
            int(baseline, 16)
            self.assertEqual(baseline, evidence.code_hash([second, first]))
            self.assertEqual(baseline, evidence.code_hash([first, second]))

            second.write_text("OTHER = 3\n", encoding="utf-8")
            self.assertNotEqual(baseline, evidence.code_hash([first, second]))

            with self.assertRaises(ValueError):
                evidence.code_hash([first, first])

    def test_code_hash_reads_workspace_relative_paths(self):
        relative = "datasets/scripts/common/review/schema.py"
        self.assertEqual(len(evidence.code_hash([relative])), 16)


class JsonTest(unittest.TestCase):
    def test_write_json_keeps_insertion_order_and_indent(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "plan.json"
            payload = {"z": 1, "a": [1, 2], "中文": "值"}
            evidence.write_json(target, payload)
            text = target.read_text(encoding="utf-8")
            self.assertIn('\n  "z": 1', text)
            self.assertIn("中文", text)
            self.assertLess(text.index('"z"'), text.index('"a"'))
            self.assertEqual(evidence.read_json(target), payload)

            target.write_text("[1, 2]\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                evidence.read_json(target)


class BatchDirTest(unittest.TestCase):
    def test_batch_dir_layout_is_not_created(self):
        with TemporaryDirectory() as tmp:
            target = evidence.batch_dir("person", "batch_x", root=tmp)
            self.assertEqual(target, Path(tmp) / "person" / "batch_x")
            self.assertFalse(target.exists())
        self.assertEqual(
            evidence.batch_dir("person", "batch_x").parts,
            ("datasets", "annotated", "person", "batch_x"),
        )
        with self.assertRaises(ValueError):
            evidence.batch_dir("", "batch_x")
        with self.assertRaises(ValueError):
            evidence.batch_dir("person", "")


def _json_line(line):
    return json.loads(line)


if __name__ == "__main__":
    unittest.main()
