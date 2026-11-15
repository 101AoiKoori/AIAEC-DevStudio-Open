"""Extract unlabelled JPEG frames from time ranges of original capture videos."""

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
SEGMENT_COLUMNS = ('segment_id', 'video', 'start_seconds', 'end_seconds', 'interval_seconds')
FRAME_COLUMNS = (
    'segment_id', 'frame', 'source_video', 'source_pts', 'source_time_base',
    'video_seconds', 'width', 'height', 'review_status', 'split',
)
VIDEO_SUFFIXES = {'.avi', '.m4v', '.mkv', '.mov', '.mp4', '.webm'}
FRAME_PATTERN = re.compile(r'^frame_(\d{6})\.jpg$')
SHOWINFO_FRAME = re.compile(
    r'Parsed_showinfo_\d+.*?\bn:\s*(\d+)\s+pts:\s*(-?\d+).*?\bs:(\d+)x(\d+)'
)
SHOWINFO_TIME_BASE = re.compile(r'Parsed_showinfo_\d+.*?config in time_base:\s*(\d+/\d+)')
MAX_FRAMES = 2500


@dataclass(frozen=True)
class Video:
    path: Path
    time_base: Fraction
    start_pts: int
    format_start: Fraction
    duration: Fraction
    stream: dict


@dataclass(frozen=True)
class Segment:
    segment_id: str
    video: Video
    start: Fraction
    end: Fraction
    interval: Fraction
    source_row: dict


def path_under(value, base):
    base = (ROOT / base).resolve()
    path = (ROOT / value).resolve()
    if not base.is_relative_to(ROOT) or not path.is_relative_to(base) or path == base:
        raise ValueError(f'路径须位于工作空间的 {base.name}/ 子目录内：{value}')
    return path


def probe_video(ffprobe, path):
    completed = subprocess.run(
        [ffprobe, '-v', 'error', '-select_streams', 'V:0', '-show_entries',
         'stream=index,codec_name,width,height,time_base,start_pts,duration_ts,'
         'duration,avg_frame_rate,nb_frames:format=start_time,duration',
         '-of', 'json', str(path)],
        capture_output=True, encoding='utf-8', errors='replace', timeout=30,
    )
    if completed.returncode:
        raise ValueError(f'{path.name} 元数据读取失败：{completed.stderr.strip()}')
    data = json.loads(completed.stdout)
    try:
        stream = data['streams'][0]
        container = data['format']
        time_base = Fraction(stream['time_base'])
        start_pts = int(stream['start_pts'])
        format_start = Fraction(container.get('start_time', '0'))
        if stream.get('duration_ts') not in (None, 'N/A'):
            duration = int(stream['duration_ts']) * time_base
        elif stream.get('duration') not in (None, 'N/A'):
            duration = Fraction(stream['duration'])
        else:
            duration = Fraction(container['duration']) + format_start - start_pts * time_base
        if min(time_base, duration, int(stream['width']), int(stream['height'])) <= 0:
            raise ValueError('nonpositive video metadata')
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f'{path.name} 缺少有效的视频起点、时间基准、尺寸或时长') from exc
    return Video(path, time_base, start_pts, format_start, duration, stream)


def seconds(value):
    value = value.strip()
    if len(value) > 24 or not re.fullmatch(r'\d+(?:\.\d+)?|\.\d+', value, re.ASCII):
        raise ValueError(f'时间须为非负十进制秒数：{value!r}')
    return Fraction(value)


def read_segments(path, raw_batch, ffprobe):
    segments, videos, ids = [], {}, set()
    total = 0
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(SEGMENT_COLUMNS).issubset(reader.fieldnames or ()):
            raise ValueError(f'CSV 必填列：{", ".join(SEGMENT_COLUMNS)}')
        for row in reader:
            try:
                if None in row or any(row[key] is None for key in SEGMENT_COLUMNS):
                    raise ValueError('CSV 行列数不符')
                segment_id = row['segment_id'].strip()
                if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', segment_id):
                    raise ValueError('segment_id 须为 1 至 64 个字母、数字、下划线或连字符，首字符须为字母或数字')
                key = segment_id.casefold()
                if key in ids or key in {'con', 'prn', 'aux', 'nul'} or re.fullmatch(r'(com|lpt)[1-9]', key):
                    raise ValueError(f'区段 ID 重复或为保留名称：{segment_id}')
                relative_video = Path(row['video'].strip().replace('\\', '/'))
                video_path = (raw_batch / relative_video).resolve()
                if (relative_video.anchor or not video_path.is_relative_to(raw_batch)
                        or video_path.suffix.lower() not in VIDEO_SUFFIXES or not video_path.is_file()):
                    raise ValueError(f'视频须是原始批次内的有效相对文件路径：{row["video"]}')
                start, end, interval = (seconds(row[column]) for column in SEGMENT_COLUMNS[2:])
                if end <= start or interval <= 0:
                    raise ValueError('结束时间须大于起始时间，抽帧间隔须大于 0')
                total += math.ceil((end - start) / interval)
                if total > MAX_FRAMES:
                    raise ValueError(f'预计抽帧超过 {MAX_FRAMES} 张，请拆分处理批次或加大间隔')
                if video_path not in videos:
                    videos[video_path] = probe_video(ffprobe, video_path)
                video = videos[video_path]
                if start >= video.duration or end > video.duration + video.time_base:
                    raise ValueError(f'区段超出视频轨道时长 {float(video.duration):.6f} 秒')
                segments.append(Segment(segment_id, video, start, end, interval, row))
                ids.add(key)
            except ValueError as exc:
                raise ValueError(f'{path.name} 第 {reader.line_num} 行：{exc}') from exc
    if not segments:
        raise ValueError('分段 CSV 没有区段')
    return segments


def frame_metadata(output, folder, log, segment, writer):
    text = log.read_text(encoding='utf-8', errors='replace')
    video = segment.video
    bases = {Fraction(value) for value in SHOWINFO_TIME_BASE.findall(text)}
    metadata = {
        int(n) + 1: (int(pts), width, height)
        for n, pts, width, height in SHOWINFO_FRAME.findall(text)
    }
    images = sorted(folder.glob('frame_*.jpg'))
    recorded = 0
    previous_pts = None
    for image in images:
        name = FRAME_PATTERN.fullmatch(image.name)
        item = metadata.get(int(name.group(1))) if name else None
        if not item or bases != {video.time_base} or image.stat().st_size == 0:
            continue
        pts, width, height = item
        position = (pts - video.start_pts) * video.time_base
        if not segment.start <= position < segment.end or (previous_pts is not None and pts <= previous_pts):
            continue
        writer.writerow({
            'segment_id': segment.segment_id, 'frame': image.relative_to(output).as_posix(),
            'source_video': video.path.relative_to(ROOT).as_posix(), 'source_pts': pts,
            'source_time_base': str(video.time_base), 'video_seconds': f'{float(position):.9f}',
            'width': width, 'height': height, 'review_status': 'pending_review', 'split': '',
        })
        previous_pts = pts
        recorded += 1
    return len(images), recorded, len(metadata)


def extract_segment(ffmpeg, output, segment, writer, record):
    video = segment.video
    first = video.start_pts + math.ceil(segment.start / video.time_base)
    end = video.start_pts + math.ceil(segment.end / video.time_base)
    step = math.ceil(segment.interval / video.time_base)
    seek = video.start_pts * video.time_base - video.format_start + segment.start - 1
    folder = output / 'frames' / segment.segment_id
    folder.mkdir(parents=True)
    log = output / 'logs' / f'{segment.segment_id}.log'
    command = [ffmpeg, '-hide_banner', '-nostdin', '-nostats', '-n', '-xerror', '-copyts']
    if seek > 0:
        command += ['-ss', f'{float(seek):.6f}']
    command += [
        '-i', str(video.path), '-map', '0:V:0',
        '-vf', f"trim=start_pts={first}:end_pts={end},"
               f"select='isnan(prev_selected_pts)+gte(pts-prev_selected_pts,{step})',showinfo=checksum=0",
        '-fps_mode', 'passthrough', '-c:v', 'mjpeg', '-q:v', '2',
        '-frames:v', str(math.ceil((segment.end - segment.start) / segment.interval) + 1),
        '-start_number', '1', str(folder / 'frame_%06d.jpg'),
    ]
    try:
        with log.open('wb') as stderr:
            completed = subprocess.run(command, stderr=stderr, timeout=1800)
        if completed.returncode:
            raise RuntimeError(f'{segment.segment_id} 的 FFmpeg 退出码为 {completed.returncode}，见 logs/{log.name}')
    finally:
        count, recorded, observed = frame_metadata(output, folder, log, segment, writer)
        record.update(images=count, frames_indexed=recorded, decoded_frames_logged=observed)
    expected_limit = math.ceil((segment.end - segment.start) / segment.interval)
    if not 0 < recorded == count == observed <= expected_limit:
        raise RuntimeError(f'{segment.segment_id} 的 JPEG 与原始时间戳未能逐帧对齐，见 logs/{log.name}')


def write_summary(output, summary):
    pending = output / 'summary.json.tmp'
    pending.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.replace(output / 'summary.json')


def extract(ffmpeg, raw_batch, segment_path, output, segments):
    """Look up registered videos and reuse a completed extraction unchanged."""
    output = path_under(output, 'datasets/annotated')
    videos = sorted({segment.video.path for segment in segments})
    with generation_guard('extract_capture_frames:' + output.relative_to(ROOT).as_posix(),
                          [segment_path, *videos], [output],
                          {'segments': [s.source_row for s in segments], 'ffmpeg': str(ffmpeg)},
                          code_paths=[Path(__file__)], procedural_inputs=[segment_path], root=ROOT) as run:
        if run.reused:
            return json.loads((output / 'summary.json').read_text(encoding='utf8'))['frames_indexed']
        if output.exists():
            shutil.rmtree(output)
        return _extract(ffmpeg, raw_batch, segment_path, output, segments)


def _extract(ffmpeg, raw_batch, segment_path, output, segments):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    videos = {segment.video.path.relative_to(ROOT).as_posix(): segment.video for segment in segments}
    summary = {
        'status': 'running',
        'raw_batch': raw_batch.relative_to(ROOT).as_posix(),
        'segment_plan': segment_path.relative_to(ROOT).as_posix(),
        'source_kind': 'camera_capture',
        'timestamp_basis': 'source_pts * source_time_base; video_seconds relative to stream start_pts',
        'sampling': 'first frame in half-open segment, then first source frame at least interval seconds later',
        'review_status': 'pending_review',
        'split': None,
        'videos': {name: {'duration_seconds': float(video.duration), 'stream': video.stream}
                   for name, video in videos.items()},
        'segments': [{'plan': segment.source_row, 'status': 'pending'} for segment in segments],
    }
    write_summary(output, summary)
    try:
        with (output / 'frames.csv').open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=FRAME_COLUMNS)
            writer.writeheader()
            for segment, record in zip(segments, summary['segments']):
                record['status'] = 'running'
                write_summary(output, summary)
                try:
                    extract_segment(ffmpeg, output, segment, writer, record)
                except (Exception, KeyboardInterrupt) as exc:
                    record['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
                    raise
                record['status'] = 'complete'
                file.flush()
                write_summary(output, summary)
                print(f'{segment.segment_id}: {record["frames_indexed"]} 张', flush=True)
        summary['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as exc:
        summary['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        summary['frames_indexed'] = sum(record.get('frames_indexed', 0) for record in summary['segments'])
        write_summary(output, summary)
    return summary['frames_indexed']


def main():
    parser = argparse.ArgumentParser(description='按区段抽取实拍原始视频，生成待审核图像及原始时间戳清单')
    parser.add_argument('--raw-batch', required=True, type=Path)
    parser.add_argument('--segments', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        raw_batch = path_under(args.raw_batch, 'datasets/raw')
        segment_path = path_under(args.segments, 'datasets/annotated')
        output = path_under(args.output, 'datasets/annotated')
        if not raw_batch.is_dir() or not segment_path.is_file():
            raise ValueError('缺少原始批次目录或分段 CSV')
        ffmpeg, ffprobe = (shutil.which(name) for name in ('ffmpeg', 'ffprobe'))
        if not ffmpeg or not ffprobe:
            raise ValueError('PATH 中须有 ffmpeg 和 ffprobe')
        segments = read_segments(segment_path, raw_batch, ffprobe)
        count = extract(ffmpeg, raw_batch, segment_path, output, segments)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(2, f'失败：{exc}\n')
    except KeyboardInterrupt:
        parser.exit(130, '抽帧中途停止。\n')
    print(f'{len(segments)} 个区段，{count} 张待审核图像。')


if __name__ == '__main__':
    main()
