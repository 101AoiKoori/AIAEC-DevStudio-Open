import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_field_capture.py"
spec = importlib.util.spec_from_file_location("evaluate_field_capture", SCRIPT)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class LabelSummaryAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original_root = subject.ROOT
        subject.ROOT = self.root

    def tearDown(self):
        subject.ROOT = self.original_root

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def records(self):
        return [
            {
                "model_key": "person",
                "image": "datasets/raw/a.png",
                "review_status": "visual_candidate",
                "image_size": [100, 100],
                "boxes": [{"class_id": 0, "xyxy": [1, 2, 20, 30]}],
            },
            {
                "model_key": "e_bike",
                "image": "datasets/raw/b.png",
                "review_status": "partial_unlabeled",
                "image_size": [100, 100],
                "boxes": [],
            },
            {
                "model_key": "e_bike",
                "image": "datasets/raw/c.png",
                "review_status": "unknown",
                "image_size": [100, 100],
                "boxes": [],
            },
        ]

    def test_counts_and_normalizes_partial_status(self):
        labels = self.write("datasets/annotated/labels.json", self.records())
        counts = subject.audit_label_files([labels])
        self.assertEqual(counts["annotation_records"], 3)
        self.assertEqual(counts["annotated_images"], 3)
        self.assertEqual(counts["known_candidate_boxes"], 1)
        self.assertEqual(counts["status_counts"], {"partial": 1, "unknown": 1, "visual_candidate": 1})

    def test_explicit_write_updates_only_derived_fields(self):
        labels = self.write("datasets/annotated/labels.json", self.records())
        summary = self.write(
            "datasets/annotated/review.json",
            {
                "annotation_records": 2,
                "annotated_images": 2,
                "known_candidate_boxes": 0,
                "status_counts": {"partial": 1, "visual_candidate": 1},
                "reviewed_records": 9,
            },
        )
        report = subject.audit_review_summary([labels], summary, write=True)
        self.assertTrue(report["mismatches"])
        updated = json.loads(summary.read_text(encoding="utf-8"))
        self.assertEqual(updated["annotation_records"], 3)
        self.assertEqual(updated["reviewed_records"], 9)
        self.assertEqual(updated["status_counts"]["partial"], 1)
        self.assertIn("count_semantics", updated)

    def test_rejects_duplicate_records_and_invalid_box_status(self):
        records = self.records()
        records.append(records[0])
        records[1]["review_status"] = "visual_candidate"
        labels = self.write("datasets/annotated/labels.json", records)
        with self.assertRaisesRegex(ValueError, "重复记录"):
            subject.audit_label_files([labels])


if __name__ == "__main__":
    unittest.main()
