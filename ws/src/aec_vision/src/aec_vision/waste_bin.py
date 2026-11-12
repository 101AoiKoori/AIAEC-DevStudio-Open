"""Waste-bin color and disposal primitives; coordinates are original-image xyxy.

The caller supplies a verified bin-body crop, model predictions and validity.
Thresholds are initial offline candidates and need camera/scene calibration.
"""
from enum import Enum
import math
from numbers import Real

import cv2
import numpy as np


class WasteCategory(str, Enum):
    RECYCLABLE = 'recyclable'
    FOOD = 'food'
    HAZARDOUS = 'hazardous'
    RESIDUAL = 'residual'
    UNKNOWN = 'unknown'


def classify_bin_color(body_bgr, min_fraction=0.20, min_margin=0.12):
    """Classify an identified bin-body BGR crop, or return UNKNOWN.

    Do not pass a full scene: neutral background also matches gray. Mask shares
    are computed over the whole crop, including white labels and highlights.
    """
    if (not isinstance(body_bgr, np.ndarray) or body_bgr.dtype != np.uint8 or
            body_bgr.ndim != 3 or body_bgr.shape[2] != 3 or body_bgr.size == 0):
        raise ValueError('expected a nonempty uint8 BGR image')
    if not (0 < min_fraction <= 1 and 0 <= min_margin <= 1):
        raise ValueError('invalid color thresholds')
    hsv = cv2.cvtColor(body_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    masks = {
        WasteCategory.RECYCLABLE: (h >= 90) & (h <= 135) & (s >= 90) & (v >= 45),
        WasteCategory.FOOD: (h >= 35) & (h <= 85) & (s >= 90) & (v >= 45),
        WasteCategory.HAZARDOUS: ((h <= 10) | (h >= 170)) & (s >= 100) & (v >= 45),
        WasteCategory.RESIDUAL: (s <= 55) & (v >= 55) & (v <= 210),
    }
    fractions = {category: float(np.mean(mask)) for category, mask in masks.items()}
    ranked = sorted(fractions, key=fractions.get, reverse=True)
    best, second = ranked[:2]
    category = best if fractions[best] >= min_fraction and fractions[best] - fractions[second] >= min_margin else WasteCategory.UNKNOWN
    return {'category': category, 'fractions': {k.value: v for k, v in fractions.items()}}


def _box(box):
    if not isinstance(box, (tuple, list, np.ndarray)) or len(box) != 4:
        raise ValueError('box must be xyxy')
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) or not math.isfinite(x) for x in box):
        raise ValueError('box coordinates must be finite numbers')
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError('box must have positive area')
    return x1, y1, x2, y2


def associate_sticker(sticker_xyxy, bins_xyxy, min_inside=0.8):
    """Return one bin index only when the sticker is sufficiently inside one box."""
    if not (0 < min_inside <= 1):
        raise ValueError('invalid inside threshold')
    x1, y1, x2, y2 = _box(sticker_xyxy)
    area = (x2 - x1) * (y2 - y1)
    candidates = []
    for index, b in enumerate(bins_xyxy):
        a, b1, c, d = _box(b)
        inside = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b1))
        if inside / area >= min_inside:
            candidates.append(index)
    return candidates[0] if len(candidates) == 1 else None


def judge_disposal(bin_category, item_key, item_mapping, item_valid, association_valid):
    """Compare a valid, uniquely associated YOLO label against generated rules.

    Callers decide model confidence/visibility validity; unseen labels or an
    ambiguous association always return unknown. Lid state is independent.
    """
    if item_valid is not True or association_valid is not True:
        return {'status': 'unknown', 'name': None, 'expected_category': None}
    try:
        category = WasteCategory(bin_category)
    except (ValueError, TypeError):
        category = WasteCategory.UNKNOWN
    entry = item_mapping.get(item_key) if isinstance(item_mapping, dict) and isinstance(item_key, str) else None
    if category == WasteCategory.UNKNOWN or not isinstance(entry, dict):
        return {'status': 'unknown', 'name': None, 'expected_category': None}
    try:
        expected = WasteCategory(entry['category'])
        name = entry['name']
        if expected == WasteCategory.UNKNOWN or not isinstance(name, str) or not name:
            raise ValueError('invalid rule')
    except (KeyError, ValueError, TypeError):
        return {'status': 'unknown', 'name': None, 'expected_category': None}
    return {'status': 'correct' if category == expected else 'incorrect',
            'name': name, 'expected_category': expected.value}
