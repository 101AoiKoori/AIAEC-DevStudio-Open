"""Deterministic checks for review sampling and similarity.

Every image here is drawn in memory with numpy/OpenCV; no workspace media is read.
"""

import unittest

import cv2
import numpy as np

from datasets.scripts.common.review import sampling
from datasets.scripts.common.review.schema import FrameRef


def textured(seed: int = 0, shape=(120, 160)) -> np.ndarray:
    """Synthetic block image: gradient background plus rectangles and light noise."""

    height, width = shape
    rows = np.linspace(30.0, 210.0, height, dtype=np.float64)[:, None]
    columns = np.linspace(0.0, 40.0, width, dtype=np.float64)[None, :]
    image = np.broadcast_to(rows + columns, (height, width)).copy()
    cv2.rectangle(image, (int(width * 0.18), int(height * 0.22)),
                  (int(width * 0.46), int(height * 0.60)), 235, -1)
    cv2.rectangle(image, (int(width * 0.58), int(height * 0.34)),
                  (int(width * 0.86), int(height * 0.76)), 25, -1)
    cv2.line(image, (0, int(height * 0.84)), (width - 1, int(height * 0.68)), 160, 3)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 6.0, (height, width))
    return np.clip(image + noise, 0.0, 255.0).astype(np.uint8)


def noisy(seed: int = 3, shape=(120, 160)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, shape, dtype=np.uint8)


def review_frames():
    """Three groups, thirty frames, including frames with no timestamp."""

    frames = []
    for index in range(12):
        frames.append(
            FrameRef(
                path="alpha_%02d.png" % index,
                parent_video="alpha.mp4",
                timestamp_seconds=float(index * 5),
                record_index=index,
                source_group="alpha",
            )
        )
    for index in range(10):
        frames.append(
            FrameRef(
                path="beta_%02d.png" % index,
                parent_video="beta.mp4",
                timestamp_seconds=float(index * 5),
                record_index=index,
            )
        )
    missing = [100.0, 200.0, None, None, 150.0, None, 50.0, 175.0]
    for index, timestamp in enumerate(missing):
        frames.append(
            FrameRef(
                path="loose_%02d.png" % index,
                parent_video=None,
                timestamp_seconds=timestamp,
                record_index=index,
            )
        )
    return frames


def group_name(frame: FrameRef) -> str:
    return frame.source_group or frame.parent_video or sampling.UNKNOWN_GROUP


class SamplingTest(unittest.TestCase):
    def test_group_by_parent_splits_three_groups_and_orders_times(self):
        frames = review_frames()
        grouped = sampling.group_by_parent(frames)
        self.assertEqual(list(grouped), ["alpha", "beta.mp4", "unknown"])
        self.assertEqual(sum(len(group) for group in grouped.values()), 30)
        self.assertEqual([len(group) for group in grouped.values()], [12, 10, 8])
        for name, group in grouped.items():
            with self.subTest(group=name):
                stamps = [frame.timestamp_seconds for frame in group]
                present = [stamp for stamp in stamps if stamp is not None]
                self.assertEqual(present, sorted(present))
                tail = stamps[len(present):]
                self.assertTrue(all(stamp is None for stamp in tail))
        unknown = grouped["unknown"]
        self.assertEqual([frame.path for frame in unknown][:5],
                         ["loose_06.png", "loose_00.png", "loose_04.png", "loose_07.png", "loose_01.png"])
        self.assertEqual([frame.path for frame in unknown][5:],
                         ["loose_02.png", "loose_03.png", "loose_05.png"])

    def test_selection_is_reproducible_for_the_same_parameters(self):
        frames = review_frames()
        first = sampling.select_representative_frames(
            frames, max_per_parent=3, min_gap_seconds=2.0, seed=11)
        second = sampling.select_representative_frames(
            frames, max_per_parent=3, min_gap_seconds=2.0, seed=11)
        self.assertEqual([frame.path for frame in first], [frame.path for frame in second])
        self.assertEqual(len(first), 9)

    def test_result_is_ordered_by_group_then_time(self):
        frames = review_frames()
        selected = sampling.select_representative_frames(
            frames, max_per_parent=4, min_gap_seconds=0.0, seed=0)
        names = [group_name(frame) for frame in selected]
        self.assertEqual(names, sorted(names))
        for name in dict.fromkeys(names):
            stamps = [frame.timestamp_seconds for frame in selected if group_name(frame) == name]
            present = [stamp for stamp in stamps if stamp is not None]
            self.assertEqual(present, sorted(present))

    def test_min_gap_drops_close_neighbours(self):
        frames = review_frames()
        gap = 12.0
        selected = sampling.select_representative_frames(
            frames, max_per_parent=10, min_gap_seconds=gap, seed=5)
        for name in ("alpha", "beta.mp4"):
            stamps = [frame.timestamp_seconds for frame in selected if group_name(frame) == name]
            self.assertTrue(stamps, name)
            for earlier, later in zip(stamps, stamps[1:]):
                self.assertGreaterEqual(later - earlier, gap)
        self.assertLess(len(selected), 30)
        # alpha ran 0..55 s every 5 s: a 12 s gap keeps at most 5 frames there.
        self.assertLessEqual(len([1 for frame in selected if group_name(frame) == "alpha"]), 5)

    def test_max_per_parent_caps_each_group(self):
        frames = review_frames()
        selected = sampling.select_representative_frames(
            frames, max_per_parent=2, min_gap_seconds=0.0, seed=0)
        counts = {}
        for frame in selected:
            counts[group_name(frame)] = counts.get(group_name(frame), 0) + 1
        self.assertEqual(counts, {"alpha": 2, "beta.mp4": 2, "unknown": 2})
        self.assertEqual(
            sampling.select_representative_frames(frames, max_per_parent=0, min_gap_seconds=1.0),
            [])

    def test_phash_matches_identical_and_separates_noise(self):
        image = textured()
        same = textured()
        self.assertEqual(sampling.phash(image), sampling.phash(same))
        self.assertEqual(sampling.hamming(sampling.phash(image), sampling.phash(same)), 0)
        self.assertGreater(
            sampling.hamming(sampling.phash(image), sampling.phash(noisy())), 8)
        self.assertEqual(sampling.hamming(0b1011, 0b0010), 2)

    def test_ssim_is_one_for_identical_and_drops_for_change(self):
        image = textured()
        self.assertAlmostEqual(sampling.ssim(image, image.copy()), 1.0, places=6)
        shifted = np.roll(image, 12, axis=1)
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        self.assertLess(sampling.ssim(image, shifted), 0.95)
        self.assertLess(sampling.ssim(image, blurred), 0.99)
        resized = cv2.resize(image, (96, 72), interpolation=cv2.INTER_AREA)
        self.assertGreater(sampling.ssim(image, resized), 0.5)
        self.assertAlmostEqual(sampling.ssim(np.zeros((32, 32), np.uint8),
                                             np.zeros((32, 32), np.uint8)), 1.0, places=6)

    def test_orb_returns_zero_when_features_are_missing(self):
        black = np.zeros((96, 96), np.uint8)
        self.assertEqual(sampling.orb_similarity(black, black), 0.0)
        self.assertEqual(sampling.orb_similarity(black, textured()), 0.0)
        self.assertGreater(sampling.orb_similarity(textured(), textured(1)), 0.0)

    def test_near_duplicates_keep_only_the_blurred_copy(self):
        image = textured()
        blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
        noise = noisy(9)
        items = [
            (FrameRef(path="clip_000.png", parent_video="clip.mp4", timestamp_seconds=0.0), image),
            (FrameRef(path="clip_001.png", parent_video="clip.mp4", timestamp_seconds=1.0), blurred),
            (FrameRef(path="clip_002.png", parent_video="clip.mp4", timestamp_seconds=2.0), noise),
        ]
        reports = sampling.find_near_duplicates(items)
        self.assertEqual(len(reports), 3)
        self.assertEqual(reports[0].group, "dup_0")
        self.assertIsNone(reports[0].duplicate_of)
        self.assertEqual(reports[0].distances, {})
        self.assertEqual(reports[1].group, "dup_0")
        self.assertEqual(reports[1].duplicate_of, "clip_000.png")
        self.assertEqual(set(reports[1].distances), {"phash", "ssim", "orb"})
        self.assertLessEqual(reports[1].distances["phash"], 8.0)
        self.assertEqual(reports[2].group, "dup_2")
        self.assertIsNone(reports[2].duplicate_of)
        self.assertEqual(reports[2].distances, {})

    def test_duplicates_are_not_claimed_across_parent_groups(self):
        image = textured()
        items = [
            (FrameRef(path="a_000.png", parent_video="a.mp4", timestamp_seconds=0.0), image),
            (FrameRef(path="b_000.png", parent_video="b.mp4", timestamp_seconds=0.0), image),
        ]
        reports = sampling.find_near_duplicates(items)
        self.assertIsNone(reports[1].duplicate_of)
        self.assertEqual(reports[1].group, "dup_1")


if __name__ == "__main__":
    unittest.main()
