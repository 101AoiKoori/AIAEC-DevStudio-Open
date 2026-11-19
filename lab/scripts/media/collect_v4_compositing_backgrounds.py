"""Collect reusable scene frames from six indexed original capture videos."""

import csv
import io
import json
import math
from collections import Counter
from contextlib import closing
from fractions import Fraction
from pathlib import Path
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_capture_frames import FRAME_COLUMNS, Segment, extract_segment, probe_video
from lab.training_common.production_guard import generation_guard
import media_index_sqlite as index


OUTPUT = ROOT / 'datasets/annotated/mixed_scene/v4_compositing_backgrounds_20261113'
GROUPS = (
    ('junction_003', 'Archive/Assets/working_sources/legacy/site/场景/三楼与场地视频/三栋楼_垃圾桶与路口_003.mp4',
     ((1, 13),), '三楼与路口平移', ('楼群', '垃圾桶', '车道线')),
    ('junction_005', 'Archive/Assets/working_sources/legacy/site/场景/三楼与场地视频/三栋楼_路口及垃圾桶_005.mp4',
     ((1, 5), (10, 21.5)), '楼群沿路口转向', ('楼群', '车辆道具', '路口线')),
    ('facades_009', 'Archive/Assets/working_sources/legacy/site/场景/三楼与场地视频/三栋楼侧向扫描_009.mp4',
     ((1, 15.5),), '楼群侧向扫描', ('楼群', '围挡', '场地边线')),
    ('mixed_012', 'Archive/Assets/working_sources/legacy/e_bike/视频/人物立牌正背面_电动车直立_倒伏和遮挡_场地扫描_012.mp4',
     ((6, 34), (47, 65), (78, 89)), '人物与电动车混合场地扫描', ('人物立牌', '直立与倒伏车', '禁行牌具', '车道线')),
    ('parking_003', 'Archive/Assets/working_sources/legacy/e_bike/视频/场地多辆电动车_直立与倒伏混合_较低视角_003.mp4',
     ((1, 12),), '低机位停车区横移', ('直立与倒伏车', '黑色围挡', '地面线')),
    ('bins_004', 'Archive/Assets/working_sources/legacy/waste_bin/视频/四色桶较低视角_灰绿闭盖_红蓝开盖_004.mp4',
     ((1.25, 13),), '低机位四色桶绕扫', ('四色桶', '开闭盖', '地面线')),
)
PER_GROUP = 20
OUTPUT_SIZE = (640, 360)
TIME_GUARD_SECONDS = .13
MIXED_012_REVIEW = ROOT / 'datasets/annotated/person/v4_video012_source_review_20261113/manifest.json'


def recorded_times(db, video):
    """Use the SQLite parent-video references to avoid known existing frames."""
    times = set()
    for raw, shared, partition in db.execute(
            'SELECT e.entry_json,r.shared_json,p.descriptor FROM evidence e '
            'JOIN partitions p ON p.name=e.partition_name '
            'LEFT JOIN manifest_rows r ON r.id=e.manifest_row_id WHERE e.path_key=?', (index.path_key(video),)):
        descriptor = json.loads(partition)
        if not descriptor['source_present'] or index.core.lifecycle_reason(descriptor['name']):
            continue
        for record in json.loads(raw)['evidence']:
            if shared is not None:
                record.update(json.loads(shared))
            if record.get('manifest', '').startswith(OUTPUT.relative_to(ROOT).as_posix() + '/'):
                continue
            claims = record.get('claims', {})
            if video not in (claims.get('source_video'), claims.get('parent_video')):
                continue
            time = claims.get('source_time_seconds', claims.get('seconds_approx'))
            if time is None:
                frame, fps = claims.get('source_frame'), claims.get('source_fps')
                if type(frame) is int and type(fps) in (int, float) and fps > 0:
                    time = frame / fps
            if type(time) in (int, float) and math.isfinite(time) and time >= 0:
                times.add(time)
    return times


def original_manifest_times(db, path, source):
    """Read fields from an indexed manifest when the cache does not extract them."""
    relative = path.relative_to(ROOT).as_posix()
    positions = {i for i, in db.execute('SELECT record_index FROM manifest_rows WHERE manifest=?', (relative,))}
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if data['source_video'] != source or positions != set(range(len(data['records']))):
        raise ValueError(f'Indexed frame selection disagrees with {relative}')
    return {row['estimated_seconds'] for row in data['records']}


def original_reference(refs):
    """Keep one existing evidence locator for the parent video when available."""
    locators = [ref for ref in refs if isinstance(ref.get('manifest'), str)
                and type(ref.get('record_index')) is int and isinstance(ref.get('relation'), str)]
    return [{key: ref[key] for key in ('manifest', 'record_index', 'relation')}
            for ref in sorted(locators, key=lambda ref: (
                ref['relation'] != 'describes', not ref['manifest'].startswith('datasets/annotated/'),
                ref['manifest'], ref['record_index']))[:1]]


def extract_group(ffmpeg, ffprobe, temp, group, excluded):
    group_id, video_path, ranges, _, _ = group
    video = probe_video(ffprobe, ROOT / video_path)
    if int(video.stream['width']) * OUTPUT_SIZE[1] != int(video.stream['height']) * OUTPUT_SIZE[0]:
        raise ValueError(f'{video_path} does not match the preserved 16:9 output')
    interval = sum(Fraction(str(end)) - Fraction(str(start)) for start, end in ranges) / 36
    frames = io.StringIO()
    writer = csv.DictWriter(frames, fieldnames=FRAME_COLUMNS)
    writer.writeheader()
    for ordinal, (start, end) in enumerate(ranges):
        segment = Segment(f'{group_id}_{ordinal}', video, Fraction(str(start)), Fraction(str(end)), interval, {})
        extract_segment(ffmpeg, temp, segment, writer, {})
    frames.seek(0)
    candidates = list(csv.DictReader(frames))
    eligible = [frame for frame in candidates
                if not any(abs(float(frame['video_seconds']) - known) <= TIME_GUARD_SECONDS for known in excluded)]
    count = min(PER_GROUP, len(eligible))
    chosen = [eligible[round(i * (len(eligible) - 1) / (count - 1))] for i in range(count)] if count > 1 else eligible
    return chosen, len(candidates), len(eligible)


def produce(ffmpeg, ffprobe, source_refs):
    with closing(index.connect(readonly=True)) as db:
        db.execute('BEGIN')
        index.validated(db, False)
        exclusions = {group[1]: recorded_times(db, group[1]) for group in GROUPS}
        mixed_video = next(group[1] for group in GROUPS if group[0] == 'mixed_012')
        exclusions[mixed_video].update(original_manifest_times(db, MIXED_012_REVIEW, mixed_video))

    with tempfile.TemporaryDirectory(prefix='v4-compositing-backgrounds-', dir=ROOT / 'tmp') as staging:
        temporary = Path(staging)
        (temporary / 'logs').mkdir()
        final = temporary / OUTPUT.name
        (final / 'images').mkdir(parents=True)
        rows, stats = [], {}
        for group in GROUPS:
            group_id, source, ranges, view, objects = group
            chosen, attempts, eligible = extract_group(ffmpeg, ffprobe, temporary, group, exclusions[source])
            if not chosen:
                raise ValueError(f'No eligible frames in {source}')
            source_records = original_reference(source_refs[source])
            for position, row in enumerate(chosen, 1):
                source_pts = int(row['source_pts'])
                seconds = float(row['video_seconds'])
                filename = f'{group_id}_{position:03d}.jpg'
                with Image.open(temporary / row['frame']) as original:
                    frame = original.convert('RGB').resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
                luma = round(ImageStat.Stat(frame.convert('L')).mean[0], 2)
                saturation = round(ImageStat.Stat(frame.convert('HSV').getchannel('S')).mean[0], 2)
                frame.save(final / 'images' / filename, quality=95, subsampling=0)
                with Image.open(final / 'images' / filename) as saved:
                    saved.load()
                    if saved.size != OUTPUT_SIZE:
                        raise ValueError(f'Unexpected output dimensions for {filename}')
                rows.append({
                    'image': (OUTPUT / 'images' / filename).relative_to(ROOT).as_posix(),
                    'source_video': source, 'source_group': source, 'background_source_group': source,
                    'source_kind': 'video_frame', 'generation_method': 'provided_video_frame',
                    'source_time_seconds': seconds, 'source_pts': source_pts,
                    'source_time_base': row['source_time_base'], 'image_size': list(OUTPUT_SIZE),
                    'image_processing': {'resize': '1920x1080 to 640x360 with preserved aspect ratio',
                                         'compression': 'FFmpeg JPEG q:v 2 intermediate, JPEG quality 95 output',
                                         'crop': 'none', 'blur': 'none'},
                    'strata': {'view_group': group_id, 'camera_view': view,
                               'source_review_objects': list(objects),
                               'mean_luma_0_255': luma, 'mean_saturation_0_255': saturation},
                    'review_status': 'source_stratum_visually_sampled', 'split': 'unassigned',
                    'label_state': 'task_labels_not_reviewed', 'source_records': source_records,
                    'source_record_scope': 'Original-video evidence locator; frame timestamp from this extraction',
                })
            stats[source] = {'sample_ranges_seconds': ranges, 'decoded_candidates': attempts,
                             'known_time_exclusions': attempts - eligible, 'retained': len(chosen)}
        keys = [(row['source_video'], row['source_pts']) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate original video timestamps in the selection')
        counts = Counter(row['source_group'] for row in rows)
        (final / 'manifest.json').write_text(json.dumps({
            'dataset_version': 'V4', 'status': 'compositing_background_candidates',
            'purpose': 'Shared real-scene backgrounds for nine V4 tasks',
            'scope': ('Existing source videos from the competition setup. Source groups are original video files. '
                      'Scene content describes representative frames in the source group. '
                      'Task-specific presence, empty-scene suitability and ground truth require per-use checks.'),
            'deduplication': f'Known indexed times excluded within {TIME_GUARD_SECONDS} seconds of candidate PTS',
            'source_groups': stats, 'counts': dict(counts), 'records': rows,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        final.rename(OUTPUT)
    return {'images': len(rows), 'source_groups': dict(counts)}


def main():
    ffmpeg, ffprobe = (shutil.which(name) for name in ('ffmpeg', 'ffprobe'))
    if not ffmpeg or not ffprobe:
        raise ValueError('ffmpeg and ffprobe are required on PATH')
    operation = 'v4_compositing_backgrounds:' + OUTPUT.relative_to(ROOT).as_posix()
    with generation_guard(operation, [ROOT / group[1] for group in GROUPS], [OUTPUT],
                          {'source_groups': GROUPS, 'per_group': PER_GROUP, 'output_size': OUTPUT_SIZE},
                          code_paths=[Path(__file__), ROOT / 'lab/scripts/media/extract_capture_frames.py'],
                          procedural_inputs=[MIXED_012_REVIEW], root=ROOT) as run:
        if not run.reused:
            if OUTPUT.exists():
                raise ValueError(f'Output already exists: {OUTPUT.relative_to(ROOT).as_posix()}')
            run.set_result(produce(ffmpeg, ffprobe, run.source_refs))
    print(json.dumps({'operation': operation, 'reused': run.reused, **run.value}, ensure_ascii=False))


if __name__ == '__main__':
    main()
