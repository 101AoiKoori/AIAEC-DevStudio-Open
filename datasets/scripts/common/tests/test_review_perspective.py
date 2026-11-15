"""Perspective templates keep reusable ground data and drop unresolved frames."""

from __future__ import annotations

import unittest

from datasets.scripts.common.review import perspective


def _payload(path, group, *, ok=True):
    return {
        "frame": {
            "path": path,
            "source_group": group,
            "parent_video": "datasets/raw/site.mp4",
            "timestamp_seconds": 1.5,
        },
        "role": "background",
        "extra": {"image_shape": [360, 640]},
        "ground": {
            "ok": ok,
            "horizon_y": 210.0,
            "polygon": [[0, 220], [639, 220], [639, 359], [0, 359]],
            "placement_patch": [[10, 300], [630, 300], [500, 260], [140, 260]],
            "foot_xy": [320, 300],
            "residual": 0.31,
            "reason_codes": ["ground_visible"] if ok else ["geometry_rejected"],
            "clearance": {"free_ratio": 0.9, "components": 1},
        },
    }


class PerspectiveTests(unittest.TestCase):
    def test_keeps_ground_and_counts_rejections(self):
        payloads = [
            _payload("datasets/annotated/site/a.png", "site"),
            _payload("datasets/annotated/site/b.png", "site", ok=False),
            {"frame": {"path": "datasets/annotated/site/c.png", "source_group": "legacy"}, "role": "background"},
        ]
        header = {"task": "person", "batch": "v4_test", "metrics": "metrics.jsonl"}
        document = perspective.build_document(header, payloads)

        self.assertEqual(document["task"], "person")
        self.assertEqual([f["path"] for f in document["frames"]],
                         ["datasets/annotated/site/a.png"])
        self.assertEqual(document["summary"]["measured"], 3)
        self.assertEqual(document["summary"]["with_ground"], 1)
        self.assertEqual(document["summary"]["without_ground"], 2)
        self.assertEqual(document["summary"]["source_groups"], {"site": 1})

    def test_template_carries_yaw_and_placement_fields(self):
        template = perspective.ground_template(_payload("datasets/annotated/site/a.png", "site"))
        self.assertEqual(template["horizon_y"], 210.0)
        self.assertEqual(template["foot_xy"], [320, 300])
        self.assertEqual(template["placement_patch"][0], [10, 300])
        self.assertEqual(template["image_shape"], [360, 640])

    def test_rejected_ground_is_not_promoted(self):
        self.assertIsNone(perspective.ground_template(_payload("x.png", "site", ok=False)))
        self.assertIsNone(perspective.ground_template({}))


if __name__ == "__main__":
    unittest.main()
