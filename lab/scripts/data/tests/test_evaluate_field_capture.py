import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_field_capture.py"
spec = importlib.util.spec_from_file_location("evaluate_field_capture", SCRIPT)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class EvaluateFieldCaptureTests(unittest.TestCase):
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

    def test_source_images_accepts_moved_capture_and_legacy_raw(self):
        archived = "Archive/Assets/captures/site_capture/onboard_camera_images/vehicle_0001.png"
        legacy = "datasets/raw/legacy.png"
        frame = "datasets/annotated/site_capture/extract_v1/frame.jpg"
        for image in (archived, legacy, frame):
            self.write(image, {})
        batch = self.root / "datasets/annotated/site_capture"
        (batch / "source_inventory.csv").write_text(
            f"media_type,current_path\nimage,{archived}\nimage,{legacy}\n", encoding="utf-8",
        )
        (batch / "extract_v1/frames.csv").write_text(
            "frame,source_video\nframe.jpg,videos/full_scene.mp4\n", encoding="utf-8",
        )
        self.assertEqual(subject.source_images(batch), {
            archived: "vehicle_camera", legacy: "vehicle_camera", frame: "full_scene",
        })

    def test_selected_images_accepts_capture_and_rejects_unrelated_archive_and_absolute_paths(self):
        archived = "Archive/Assets/captures/car_real_capture/P02_PS1.png"
        original = "Archive/Assets/CarRealPicture/P02_PS1.png"
        raw = "datasets/raw/source.png"
        annotated = "datasets/annotated/batch/frame.jpg"
        unrelated = "Archive/Assets/working_sources/reference/other.png"
        selected = (archived, original, raw, annotated)
        for image in (*selected, unrelated):
            self.write(image, {})
        path = "datasets/annotated/selection.json"
        manifest = {"batch_id": "waypoints", "items": [{"path": item} for item in selected]}
        self.assertEqual(subject.selected_images(self.write(path, manifest)), {
            image: "waypoints" for image in selected
        })
        for item in (unrelated, str(self.root / archived)):
            with self.subTest(item=item), self.assertRaises(ValueError):
                subject.selected_images(self.write(path, {"batch_id": "waypoints", "items": [{"path": item}]}))

    def test_score_reuses_frozen_result_only_for_verified_file_copy(self):
        archived = "Archive/Assets/captures/car_real_capture/P02_PS1.png"
        original = "Archive/Assets/CarRealPicture/P02_PS1.png"
        image = self.write(archived, {})
        self.write(original, {})
        copy = {"path": archived, "original_path": original, "generation_method": "file_copy"}
        declarations = self.write(subject.SOURCE_COPY_MANIFEST, copy)
        self.write("runs/frozen/prediction_summary.json", {
            "models": {"person": {"status": "complete", "class_names": ["community"]}},
        })
        self.write("runs/frozen/predictions/person.jsonl", {
            "image": original, "source": "waypoints", "shape_hw": [100, 100], "boxes": [],
        })
        labels = self.write("datasets/annotated/current.json", [{
            "model_key": "person", "image": archived, "review_status": "visual_negative", "boxes": [],
        }])
        run = self.root / "runs/frozen"
        out = self.root / "runs/report"
        self.assertEqual(subject.score(run, [labels], out, .25), 0)
        (matched,) = subject.read_json(out / "per_image.json")
        self.assertEqual(matched["image"], archived)
        self.assertEqual(matched["prediction_image"], original)
        self.assertEqual(matched["prediction_match"]["manifest"], subject.SOURCE_COPY_MANIFEST)

        declarations.write_text(
            json.dumps({**copy, "generation_method": "lossless_format_conversion"}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "缺少唯一的冻结推理结果"):
            subject.score(run, [labels], self.root / "runs/converted", .25)
        declarations.write_text(json.dumps(copy) + "\n", encoding="utf-8")
        image.write_bytes(b"different file content")
        with self.assertRaisesRegex(ValueError, "缺少唯一的冻结推理结果"):
            subject.score(run, [labels], self.root / "runs/mismatched", .25)


if __name__ == "__main__":
    unittest.main()
