"""Prepare unlabelled, source-grouped e-bike video frames for manual review."""
import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
VIDEO_SUFFIXES = {'.mp4', '.mov', '.m4v', '.avi', '.mkv'}


def workspace_path(value):
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError('路径必须位于工作空间内') from exc
    return path


def save_manifest(output, records, fps):
    manifest = {
        'purpose': 'unlabelled review candidates; no train/validation/test assignment',
        'frame_sampling_fps': fps,
        'grouping': 'frames from one video stay together; relationships between videos are unknown',
        'status': 'pending_manual_review',
        'sources': records,
    }
    (output / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def write_gallery(output, records):
    parts = ['<!doctype html><meta charset="utf-8"><title>电动车视频抽帧复核</title>',
             '<style>body{font:16px sans-serif;margin:2rem;background:#f7f7f7}'
             '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:1rem}'
             'figure{margin:0;background:white;padding:.5rem}img{width:100%;height:145px;object-fit:contain}'
             'figcaption{overflow-wrap:anywhere}</style>',
             '<h1>电动车视频抽帧复核</h1><p>全部图片待人工标注和复核；尚未划分训练、验证或测试集。</p>']
    for record in records:
        parts.append('<h2>' + html.escape(record['source_id'] + ' — ' + record['source']) + '</h2>')
        parts.append('<p>处理状态：' + html.escape(record['status']) + '</p><div class="grid">')
        for frame in record['frames']:
            safe = html.escape(frame, quote=True)
            parts.append(f'<figure><a href="{safe}"><img loading="lazy" src="{safe}"></a>'
                         f'<figcaption>{html.escape(Path(frame).name)}</figcaption></figure>')
        parts.append('</div>')
    (output / 'review.html').write_text('\n'.join(parts), encoding='utf-8')


def prepare(videos, output, fps):
    with generation_guard('e_bike.prepare_sources:' + str(output.resolve()), videos, [output],
                          {'fps': fps, 'videos': [str(x) for x in videos]},
                          code_paths=[Path(__file__)]) as run:
        if run.reused:
            return json.loads((output / 'manifest.json').read_text(encoding='utf-8'))['sources']
        return _prepare(videos, output, fps)


def _prepare(videos, output, fps):
    if output.exists():
        raise ValueError('输出目录已存在，请选择新目录')
    if not 0 < fps <= 2:
        raise ValueError('抽帧频率须大于 0 且不超过 2 帧/秒')
    if not videos:
        raise ValueError('至少选择一段视频')
    if len(set(videos)) != len(videos):
        raise ValueError('视频路径重复')
    for video in videos:
        if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f'视频不存在或格式不支持: {video}')
    output.mkdir(parents=True)
    records = []
    save_manifest(output, records, fps)
    try:
        for index, video in enumerate(videos, 1):
            source_id = f'source_{index:03d}'
            dest = output / 'images' / source_id
            dest.mkdir(parents=True)
            record = {'source_id': source_id, 'source': video.relative_to(ROOT).as_posix(),
                      'acquisition_relationship': 'unknown', 'split': None,
                      'status': 'extracting', 'frames': []}
            records.append(record)
            save_manifest(output, records, fps)
            try:
                proc = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(video),
                     '-vf', f'fps={fps}', '-fps_mode', 'vfr', '-q:v', '2', str(dest / 'frame_%06d.jpg')],
                    capture_output=True, text=True, encoding='utf-8', errors='replace', check=False)
                record['frames'] = [p.relative_to(output).as_posix() for p in sorted(dest.glob('frame_*.jpg'))]
                if proc.returncode != 0 or not record['frames']:
                    record['status'] = 'failed'
                    record['error'] = proc.stderr.strip() or '未提取到帧'
                    raise RuntimeError(f'{video.name} 解码失败: {record["error"]}')
                record['status'] = 'pending_manual_review'
            except OSError as exc:
                record['status'] = 'failed'
                record['error'] = str(exc)
                raise
            finally:
                save_manifest(output, records, fps)
    finally:
        write_gallery(output, records)
    return records


def main():
    parser = argparse.ArgumentParser(description='抽帧并生成未标注素材复核页面')
    parser.add_argument('--video', action='append', required=True, help='工作空间内的视频，可重复指定')
    parser.add_argument('--output', required=True, help='不存在的工作空间内输出目录')
    parser.add_argument('--fps', type=float, default=0.5)
    args = parser.parse_args()
    try:
        records = prepare([workspace_path(p) for p in args.video], workspace_path(args.output), args.fps)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'失败：{exc}\n')
    print(f'处理 {len(records)} 段视频，共 {sum(len(r["frames"]) for r in records)} 帧；未标注、未切分')


if __name__ == '__main__':
    main()
