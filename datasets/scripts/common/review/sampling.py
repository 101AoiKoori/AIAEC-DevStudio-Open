"""Deterministic frame sampling and perceptual similarity for candidate review.

Pure module-level functions with explicit parameters: no file IO, no global
random state, no task-specific thresholds. Records come from :mod:`.schema`, so
plans and metrics stay comparable across tasks.

The similarity measures answer one question only: are two frames the same
observation? They do not establish that a frame is usable. ``ssim`` and
``orb_similarity`` accept 2-D grayscale arrays (a 3-channel BGR/BGRA array is
converted) and never write to their inputs.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .schema import DuplicateReport, FrameRef

__all__ = [
    "UNKNOWN_GROUP",
    "group_by_parent",
    "select_representative_frames",
    "phash",
    "hamming",
    "ssim",
    "orb_similarity",
    "find_near_duplicates",
]

UNKNOWN_GROUP = "unknown"

# DCT domain is sampled at 4x the hash size, the usual perceptual-hash baseline.
_DCT_SIZE_FACTOR = 4
# Gaussian window for SSIM, matching the reference 11x11, sigma 1.5 setting.
_SSIM_WINDOW_SIGMA = 1.5
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03
# Lowe ratio for ORB descriptor matching.
_ORB_RATIO = 0.75


def _as_gray(value: np.ndarray) -> np.ndarray:
    """Return a 2-D float64 grayscale image scaled to 0..255."""

    array = np.asarray(value)
    if array.ndim == 3:
        channels = array.shape[2]
        if channels == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        elif channels >= 1:
            array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError("expected a 2-D grayscale image")
    if array.size == 0:
        raise ValueError("expected a non-empty image")
    if array.dtype == np.bool_:
        return array.astype(np.float64) * 255.0
    work = np.nan_to_num(array.astype(np.float64), nan=0.0, posinf=255.0, neginf=0.0)
    if float(work.max()) <= 1.0:
        work = work * 255.0
    return np.clip(work, 0.0, 255.0)


def _as_uint8(value: np.ndarray) -> np.ndarray:
    """Return a contiguous 2-D uint8 grayscale image."""

    return np.ascontiguousarray(np.clip(_as_gray(value), 0.0, 255.0).astype(np.uint8))


def _group_key(frame: FrameRef) -> str:
    return frame.source_group or frame.parent_video or UNKNOWN_GROUP


def _time_key(frame: FrameRef) -> Tuple[int, float, str]:
    if frame.timestamp_seconds is None:
        return (1, 0.0, frame.path)
    return (0, float(frame.timestamp_seconds), frame.path)


def group_by_parent(frames: Sequence[FrameRef]) -> dict:
    """Group frames by ``source_group``, else ``parent_video``, else ``unknown``.

    Groups are returned in name order and every group is ordered by timestamp
    with missing timestamps last, then by path, so two runs over the same input
    always produce the same structure.
    """

    grouped = {}
    for frame in frames:
        grouped.setdefault(_group_key(frame), []).append(frame)
    return {name: sorted(grouped[name], key=_time_key) for name in sorted(grouped)}


def _thin_by_gap(group: Sequence[FrameRef], gap: float) -> List[FrameRef]:
    """Greedily drop frames closer than ``gap`` seconds to the last kept frame.

    A frame without a timestamp cannot be shown to be near its neighbours, so it
    is kept and does not anchor the gap for later frames.
    """

    kept: List[FrameRef] = []
    last_timestamp: Optional[float] = None
    for frame in group:
        if frame.timestamp_seconds is None:
            kept.append(frame)
            continue
        value = float(frame.timestamp_seconds)
        if last_timestamp is not None and abs(value - last_timestamp) < gap:
            continue
        kept.append(frame)
        last_timestamp = value
    return kept


def _uniform_indices(count: int, limit: int, seed: int) -> List[int]:
    """Pick at most ``limit`` evenly spaced indices out of ``count``.

    ``seed`` only breaks ties created by rounding two target positions onto the
    same index; a local ``random.Random`` instance keeps the caller's and the
    interpreter's random state untouched.
    """

    if limit <= 0 or count <= 0:
        return []
    if count <= limit:
        return list(range(count))
    rng = None
    used = set()
    chosen: List[int] = []
    for position in np.linspace(0.0, float(count - 1), limit):
        target = float(position)
        index = min(max(int(round(target)), 0), count - 1)
        if index in used:
            free = [candidate for candidate in range(count) if candidate not in used]
            free.sort(key=lambda candidate: (abs(candidate - target), candidate))
            closest = abs(free[0] - target)
            tied = [candidate for candidate in free if abs(candidate - target) == closest]
            if len(tied) > 1:
                if rng is None:
                    rng = random.Random(seed)
                index = tied[rng.randrange(len(tied))]
            else:
                index = tied[0]
        used.add(index)
        chosen.append(index)
    return sorted(chosen)


def select_representative_frames(
    frames: Sequence[FrameRef],
    *,
    max_per_parent: int,
    min_gap_seconds: float,
    seed: int = 0,
) -> List[FrameRef]:
    """Choose a small, reproducible subset of frames per parent group.

    Each group is thinned by ``min_gap_seconds`` first (time order, a frame is
    skipped when it is closer than the gap to the last kept frame), then spread
    evenly down to ``max_per_parent`` frames. The result is ordered by group name
    and then by time. A non-positive ``max_per_parent`` selects nothing; a
    negative or non-finite gap is treated as ``0``.
    """

    limit = int(max_per_parent)
    if limit <= 0:
        return []
    gap = float(min_gap_seconds)
    if not math.isfinite(gap) or gap < 0.0:
        gap = 0.0
    selected: List[FrameRef] = []
    for name, group in group_by_parent(frames).items():
        thinned = _thin_by_gap(group, gap)
        for index in _uniform_indices(len(thinned), limit, int(seed)):
            selected.append(thinned[index])
    return selected


def _dct_matrix(size: int) -> np.ndarray:
    """Orthonormal DCT-II matrix, so ``matrix @ block @ matrix.T`` transforms."""

    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(np.pi * (2.0 * positions[None, :] + 1.0) * frequencies / (2.0 * size))
    matrix *= math.sqrt(2.0 / size)
    matrix[0] /= math.sqrt(2.0)
    return matrix


def phash(gray: np.ndarray, hash_size: int = 8) -> int:
    """DCT perceptual hash of a grayscale image as an integer bit pattern.

    The image is resized to ``hash_size * 4`` square, DCT-transformed and the
    low-frequency ``hash_size`` square is compared to its median, dropping the DC
    coefficient. Bits are packed least-significant first. Pure numpy: OpenCV's
    ``img_hash`` module is not guaranteed to be built.
    """

    size = int(hash_size)
    if size < 1:
        raise ValueError("hash_size must be positive")
    work = max(1, size * _DCT_SIZE_FACTOR)
    base = _as_uint8(gray)
    resized = cv2.resize(base, (work, work), interpolation=cv2.INTER_AREA).astype(np.float64)
    matrix = _dct_matrix(work)
    coefficients = matrix @ resized @ matrix.T
    values = coefficients[:size, :size].flatten()[1:]
    if values.size == 0:
        return 0
    bits = values > float(np.median(values))
    code = 0
    for index, bit in enumerate(bits):
        if bit:
            code |= 1 << index
    return code


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two integer hashes."""

    return bin(int(a) ^ int(b)).count("1")


def _aligned_pair(a_gray: np.ndarray, b_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two same-size float64 images, resized to the smaller shape."""

    left = _as_gray(a_gray)
    right = _as_gray(b_gray)
    if left.shape != right.shape:
        height = min(left.shape[0], right.shape[0])
        width = min(left.shape[1], right.shape[1])
        left = cv2.resize(left, (width, height), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (width, height), interpolation=cv2.INTER_AREA)
    return left, right


def ssim(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    """Mean structural similarity over a Gaussian window, clamped to [-1, 1].

    Images of different sizes are resized to the smaller height and width before
    comparison; a constant image still scores through the stabilising constants.
    """

    left, right = _aligned_pair(a_gray, b_gray)
    c1 = (_SSIM_K1 * 255.0) ** 2
    c2 = (_SSIM_K2 * 255.0) ** 2

    def blur(image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (0, 0), _SSIM_WINDOW_SIGMA, borderType=cv2.BORDER_REFLECT)

    mu_left = blur(left)
    mu_right = blur(right)
    left_sq = mu_left * mu_left
    right_sq = mu_right * mu_right
    cross = mu_left * mu_right
    var_left = blur(left * left) - left_sq
    var_right = blur(right * right) - right_sq
    covariance = blur(left * right) - cross
    numerator = (2.0 * cross + c1) * (2.0 * covariance + c2)
    denominator = (left_sq + right_sq + c1) * (var_left + var_right + c2)
    return float(np.clip(np.mean(numerator / denominator), -1.0, 1.0))


def orb_similarity(
    a_gray: np.ndarray,
    b_gray: np.ndarray,
    *,
    max_features: int = 500,
) -> float:
    """Ratio-test ORB matches divided by the smaller keypoint count.

    Returns ``0.0`` when either image yields fewer than two descriptors: a single
    keypoint carries no ratio test and must not be reported as a match.
    """

    features = int(max_features)
    if features < 2:
        return 0.0
    detector = cv2.ORB_create(nfeatures=features)
    left = _as_uint8(a_gray)
    right = _as_uint8(b_gray)
    left_keys, left_desc = detector.detectAndCompute(left, None)
    if left_desc is None or len(left_keys) < 2:
        return 0.0
    right_keys, right_desc = detector.detectAndCompute(right, None)
    if right_desc is None or len(right_keys) < 2:
        return 0.0
    smaller = min(len(left_keys), len(right_keys))
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(left_desc, right_desc, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < _ORB_RATIO * pair[1].distance]
    return float(min(1.0, len(good) / float(smaller)))


def find_near_duplicates(
    items: Sequence[Tuple[FrameRef, np.ndarray]],
    *,
    phash_max_distance: int = 8,
    ssim_min: float = 0.95,
    orb_min: float = 0.35,
    require_two: bool = True,
) -> List[DuplicateReport]:
    """Flag frames that repeat the same observation inside one parent group.

    ``items`` pairs a frame with its grayscale array. Frames are compared in input
    order and only against frames of the same parent group: frames from different
    groups are never the same observation. A pair is a duplicate when at least
    ``2`` metrics hit (``require_two``) or at least ``1`` when it is ``False``.

    The first frame of a cluster is its representative and gets
    ``duplicate_of=None`` with empty ``distances``: nothing was measured for it.
    Members carry ``group="dup_<representative index>"`` and their distances to
    the representative. The result has one report per input, in input order.
    """

    entries = list(items)
    reports: List[Optional[DuplicateReport]] = [None] * len(entries)
    hashes = [phash(gray) for _, gray in entries]
    required_hits = 2 if require_two else 1
    for start in range(len(entries)):
        if reports[start] is not None:
            continue
        frame, gray = entries[start]
        key = _group_key(frame)
        reports[start] = DuplicateReport(group="dup_%d" % start, duplicate_of=None, distances={})
        for other in range(start + 1, len(entries)):
            if reports[other] is not None:
                continue
            other_frame, other_gray = entries[other]
            if _group_key(other_frame) != key:
                continue
            distances = {
                "phash": float(hamming(hashes[start], hashes[other])),
                "ssim": ssim(gray, other_gray),
                "orb": orb_similarity(gray, other_gray),
            }
            hits = int(distances["phash"] <= phash_max_distance)
            hits += int(distances["ssim"] >= ssim_min)
            hits += int(distances["orb"] >= orb_min)
            if hits < required_hits:
                continue
            reports[other] = DuplicateReport(
                group="dup_%d" % start,
                duplicate_of=frame.path,
                distances=distances,
            )
    return [report for report in reports if report is not None]
