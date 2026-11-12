"""Single-frame HSV traffic-light candidates for calibrated camera regions.

The three normalized xyxy regions, HSV ranges and minimum component areas must
be supplied from calibration for the current signal location. OpenCV HSV uses
H 0..179, S/V 0..255; red may need two ranges. Results are observations only,
not a motion authorization: callers must also establish position, frame age,
temporal stability and safe navigation behavior.
"""
import math
from numbers import Real

import cv2
import numpy as np


COLORS = ('red', 'yellow', 'green')


def _finite(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, Real) and math.isfinite(value)


def _validate_config(regions, hsv_ranges, min_areas):
    if any(not isinstance(mapping, dict) or set(mapping) != set(COLORS)
           for mapping in (regions, hsv_ranges, min_areas)):
        raise ValueError('configuration must contain exactly red, yellow and green')
    for color in COLORS:
        roi = regions[color]
        if not isinstance(roi, (list, tuple)) or len(roi) != 4 or not all(_finite(v) for v in roi):
            raise ValueError('each region must be normalized xyxy')
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('regions must have positive area within the image')
        area = min_areas[color]
        if isinstance(area, (bool, np.bool_)) or not isinstance(area, (int, np.integer)) or area < 1:
            raise ValueError('minimum areas must be positive pixel counts')
        ranges = hsv_ranges[color]
        if not isinstance(ranges, (tuple, list)) or not ranges:
            raise ValueError('each color needs at least one HSV range')
        for pair in ranges:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError('HSV range must have lower and upper triples')
            lo, hi = pair
            if not isinstance(lo, (tuple, list)) or not isinstance(hi, (tuple, list)) or len(lo) != 3 or len(hi) != 3:
                raise ValueError('HSV bounds must be triples')
            for lower, upper, maximum in zip(lo, hi, (179, 255, 255)):
                if (isinstance(lower, (bool, np.bool_)) or isinstance(upper, (bool, np.bool_)) or
                        not isinstance(lower, (int, np.integer)) or not isinstance(upper, (int, np.integer)) or
                        not 0 <= lower <= upper <= maximum):
                    raise ValueError('invalid OpenCV HSV bounds')


def detect_traffic_light(frame_bgr, regions, hsv_ranges, min_areas):
    """Return one single-frame color candidate, or unknown for ambiguous input.

    regions maps each color to its normalized xyxy image rectangle. Each region
    can extend across the image in either direction. hsv_ranges maps colors to
    lists of inclusive (lower, upper) HSV triples; min_areas maps colors to the
    smallest connected component area in pixels. All must be calibrated.
    """
    _validate_config(regions, hsv_ranges, min_areas)
    unknown = {'status': 'unknown', 'color': 'unknown', 'areas': {}}
    if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8 or
            frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.size == 0):
        return unknown
    height, width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    areas = {}
    for color in COLORS:
        x1, y1, x2, y2 = regions[color]
        x1, y1 = math.floor(x1 * width), math.floor(y1 * height)
        x2, y2 = math.ceil(x2 * width), math.ceil(y2 * height)
        crop = hsv[y1:y2, x1:x2]
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        for lo, hi in hsv_ranges[color]:
            mask = cv2.bitwise_or(mask, cv2.inRange(crop, np.array(lo, dtype=np.uint8),
                                                    np.array(hi, dtype=np.uint8)))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas[color] = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    matches = [color for color in COLORS if areas[color] >= min_areas[color]]
    if len(matches) != 1:
        return {'status': 'unknown', 'color': 'unknown', 'areas': areas}
    return {'status': 'candidate', 'color': matches[0], 'areas': areas}


class TrafficLightState:
    """Confirm same-color frames for one location and expire them on query.

    Timestamps must be from the same monotonic clock. The caller must query with
    the current time even during camera failure and provide a trustworthy
    stopped-at-the-observation-point flag. This state does not drive a robot.
    """

    def __init__(self, required_frames, max_age):
        if isinstance(required_frames, bool) or not isinstance(required_frames, int) or required_frames < 1:
            raise ValueError('required_frames must be a positive integer')
        if not _finite(max_age) or max_age <= 0:
            raise ValueError('max_age must be positive and finite')
        self.required_frames = required_frames
        self.max_age = max_age
        self._last_frame_time = None
        self._clear()

    def _clear(self):
        self._color = None
        self._count = 0
        self._time = None

    def read(self, now, positioned):
        if (positioned is not True or not _finite(now) or self._time is None or
                now < self._time or now - self._time > self.max_age):
            self._clear()
            return {'status': 'unknown', 'color': 'unknown'}
        if self._count < self.required_frames:
            return {'status': 'unknown', 'color': 'unknown'}
        return {'status': 'stable', 'color': self._color, 'observed_at': self._time}

    def update(self, observation, frame_time, now, positioned):
        timestamp_ok = (_finite(frame_time) and _finite(now) and frame_time <= now and
                        (self._last_frame_time is None or frame_time > self._last_frame_time))
        if timestamp_ok:
            self._last_frame_time = frame_time
        valid = (positioned is True and _finite(frame_time) and _finite(now) and
                 frame_time <= now and now - frame_time <= self.max_age and
                 isinstance(observation, dict) and observation.get('status') == 'candidate' and
                 observation.get('color') in COLORS)
        if not valid or not timestamp_ok:
            self._clear()
            return {'status': 'unknown', 'color': 'unknown'}
        color = observation['color']
        if self._time is None or frame_time - self._time > self.max_age or color != self._color:
            self._color, self._count = color, 1
        else:
            self._count += 1
        self._time = frame_time
        return self.read(now, positioned)
