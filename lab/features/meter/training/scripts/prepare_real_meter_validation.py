"""Extract a single real-video group and propose annotations for review.

The reference coordinates apply only to the video recorded in this workspace.
Projected labels and detected ticks are candidates, not reviewed ground truth.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard


TIMES = (2.0, 3.0, 5.0, 8.7, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0)
# Pixel coordinates on the decoded, portrait-oriented 9 s image.
BOX = [700, 580, 1765, 1790]
POINTS = {
    "temperature_pivot": [1171, 1227],
    "temperature_tip": [1445, 860],
    "humidity_pivot": [1196, 1488],
    "humidity_tip": [1209, 1647],
    "temperature_min": [900, 1420],
    "temperature_max": [1547, 1430],
    "humidity_min": [1298, 1365],
    "humidity_max": [1092, 1364],
}


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot read extracted frame: " + str(path))
    return image


def write_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError("Cannot encode " + str(path))
    encoded.tofile(str(path))


def project(points, matrix):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, matrix).reshape(-1, 2).tolist()


def candidate_ticks(reference):
    """Detect dark radial line segments; candidates can include noise/duplicates."""
    x1, y1, x2, y2 = BOX
    crop = reference[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 110)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, 10,
                            minLineLength=7, maxLineGap=3)
    candidates = []
    if lines is None:
        return candidates
    for x0, y0, x3, y3 in lines[:, 0]:
        a = np.array([x0 + x1, y0 + y1], dtype=float)
        b = np.array([x3 + x1, y3 + y1], dtype=float)
        mid = (a + b) / 2
        vector = b - a
        length = np.linalg.norm(vector)
        for name, center, low, high in (
            ("temperature", POINTS["temperature_pivot"], 340, 540),
            ("humidity", POINTS["humidity_pivot"], 110, 190),
        ):
            radius = mid - np.array(center)
            distance = np.linalg.norm(radius)
            if (low < distance < high and 7 <= length <= 100
                    and abs(np.dot(radius, vector)) / (distance * length) > 0.94
                    and (name != "temperature" or mid[1] < 1470)
                    and (name != "humidity" or mid[1] > 1350)):
                candidates.append({"scale": name, "line": [a.tolist(), b.tolist()],
                                   "value": None, "status": "candidate_unreviewed"})
                break
    return candidates


def find_homography(reference, frame):
    scale = 0.5
    ref = cv2.resize(reference, None, fx=scale, fy=scale)
    dst = cv2.resize(frame, None, fx=scale, fy=scale)
    mask = np.zeros(ref.shape[:2], np.uint8)
    x1, y1, x2, y2 = (int(v * scale) for v in BOX)
    mask[y1:y2, x1:x2] = 255
    sift = cv2.SIFT_create(nfeatures=7000)
    kr, dr = sift.detectAndCompute(ref, mask)
    kf, df = sift.detectAndCompute(dst, None)
    if dr is None or df is None:
        return None, 0, None
    matches = cv2.BFMatcher().knnMatch(dr, df, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.7 * n.distance]
    if len(good) < 12:
        return None, 0, None
    source = np.float32([kr[m.queryIdx].pt for m in good])
    target = np.float32([kf[m.trainIdx].pt for m in good])
    mat, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.0)
    if mat is None:
        return None, len(good), None
    if int(inliers.sum()) < 30:
        return None, len(good), int(inliers.sum())
    s = np.diag([scale, scale, 1.0])
    return np.linalg.inv(s) @ mat @ s, len(good), int(inliers.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.video, args.output = args.video.resolve(), args.output.resolve()
    with generation_guard('meter.prepare_real_meter_validation:' + str(args.output.resolve()),
                          [args.video], [args.output], {'times': TIMES},
                          code_paths=[Path(__file__)]) as run:
        if run.reused:
            print('REUSED', args.output)
            return
        _produce(args, run.source_refs)


def _produce(args, source_refs):
    args.output.mkdir(parents=True, exist_ok=False)
    for sub in ("images", "annotations", "previews"):
        (args.output / sub).mkdir()
    for t in TIMES:
        name = f"frame_{int(t * 1000):06d}"
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(args.video),
                        "-frames:v", "1", "-y", str(args.output / "images" / (name + ".png"))], check=True)
    reference = read_image(args.output / "images" / "frame_009000.png")
    if reference.shape[:2] != (3840, 2160):
        raise ValueError("Reference coordinates require a 2160 x 3840 portrait decode")
    ticks = candidate_ticks(reference)
    records = []
    for t in TIMES:
        name = f"frame_{int(t * 1000):06d}"
        frame = read_image(args.output / "images" / (name + ".png"))
        if t == 9.0:
            matrix, matches, inliers = np.eye(3), None, None
        else:
            matrix, matches, inliers = find_homography(reference, frame)
        record = {"id": name, "seconds": t, "source_group": args.video.name,
                  "split": "validation_candidate", "status": "unreviewed",
                  "reading": {"temperature_c": None, "humidity_percent": None},
                  "homography_matches": matches, "homography_inliers": inliers,
                  "box_polygon": None, "keypoints": {}, "ticks": []}
        if matrix is None:
            record["status"] = ("registration_rejected" if inliers is not None
                                else "registration_failed")
        if matrix is not None:
            x1, y1, x2, y2 = BOX
            polygon = project([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], matrix)
            record["box_polygon"] = polygon
            pp = project(list(POINTS.values()), matrix)
            record["keypoints"] = {k: {"xy": p, "status": "candidate_unreviewed"}
                                   for k, p in zip(POINTS, pp)}
            for tick in ticks:
                record["ticks"].append({**tick, "line": project(tick["line"], matrix)})
            preview = frame.copy()
            cv2.polylines(preview, [np.array(polygon, np.int32)], True, (255, 255, 0), 4)
            for tick in record["ticks"]:
                a, b = np.array(tick["line"], np.int32)
                cv2.line(preview, tuple(a), tuple(b), (0, 255, 0), 3)
            for prefix, color in (("temperature", (255, 0, 255)), ("humidity", (255, 128, 0))):
                a = tuple(np.array(record["keypoints"][prefix + "_pivot"]["xy"], np.int32))
                b = tuple(np.array(record["keypoints"][prefix + "_tip"]["xy"], np.int32))
                cv2.line(preview, a, b, color, 5)
            for key, p in record["keypoints"].items():
                xy = tuple(np.array(p["xy"], np.int32))
                cv2.circle(preview, xy, 9, (0, 0, 255), -1)
                cv2.putText(preview, key, (xy[0] + 8, xy[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(preview, "CANDIDATE - REQUIRES REVIEW", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            write_image(args.output / "previews" / (name + ".jpg"), preview)
        (args.output / "annotations" / (name + ".json")).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append({k: record[k] for k in ("id", "seconds", "status", "homography_matches", "homography_inliers")})
    (args.output / "manifest.json").write_text(json.dumps({
        "source": str(args.video), "source_group": args.video.name,
        "input_source_refs": source_refs,
        "description": "One capture session and one fixed needle state; all labels are unreviewed candidates.",
        "reference": "frame_009000", "tick_candidates_reference": len(ticks),
        "frames": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(records), "tick_candidates": len(ticks), "records": records}, ensure_ascii=False))


if __name__ == "__main__":
    main()
