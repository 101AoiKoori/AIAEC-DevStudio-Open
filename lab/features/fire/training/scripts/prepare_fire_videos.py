"""Extract grouped fire-task videos into an unlabelled review set."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}


def workspace_path(text):
    path = (ROOT / text).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError('路径必须位于工作空间内') from exc
    return path


def prepare(source, output, fps):
    if not source.is_dir():
        raise ValueError('视频目录不存在')
    if not 0 < fps <= 10:
        raise ValueError('抽帧频率必须大于 0 且不超过 10 帧/秒')
    groups = []
    for split in ('train', 'val', 'test'):
        folder = source / split
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in SUFFIXES:
                    if path.is_symlink():
                        raise ValueError('视频不能是符号链接')
                    groups.append((split, path))
    if not groups:
        raise ValueError('train、val、test 目录中没有视频')
    names = [p.name.casefold() for _, p in groups]
    if len(names) != len(set(names)):
        raise ValueError('视频文件名重复；请为各采集视频使用不同名称')
    with generation_guard('fire.prepare_fire_videos:' + output.relative_to(ROOT).as_posix(),
                          [path for _, path in groups], [output],
                          {'fps': fps, 'groups': [(split, path.relative_to(ROOT).as_posix())
                                                 for split, path in groups]},
                          code_paths=[Path(__file__)], root=ROOT) as run:
        if run.reused:
            return json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        return _prepare_groups(groups, output, fps, run.source_refs)


def _prepare_groups(groups, output, fps, source_refs):
    output.mkdir(parents=True)
    manifest = []
    for index, (split, path) in enumerate(groups):
        video_id = f'video_{index + 1:04d}'
        dest = output / 'images' / split / video_id
        dest.mkdir(parents=True)
        proc = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                               '-i', str(path), '-vf', f'fps={fps}', '-fps_mode:v', 'vfr',
                               str(dest / 'frame_%06d.png')], capture_output=True, text=True)
        frames = sorted(dest.glob('frame_*.png'))
        record = {'video_id': video_id, 'split': split, 'source': path.relative_to(ROOT).as_posix(),
                  'fps': fps, 'frames': len(frames), 'status': 'pending_review',
                  'source_references': source_refs[path.relative_to(ROOT).as_posix()]}
        manifest.append(record)
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        if proc.returncode or not frames:
            raise RuntimeError(f'{path.name} 解码失败，保留已提取结果供检查：{proc.stderr.strip()}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='按视频分组抽帧，结果需要标注复核')
    parser.add_argument('--source', default='Archive/Assets/captures/fire/videos')
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=float, default=1.0)
    args = parser.parse_args()
    try:
        records = prepare(workspace_path(args.source), workspace_path(args.output), args.fps)
        print(f'处理 {len(records)} 段视频，抽取 {sum(x["frames"] for x in records)} 帧；等待标注')
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'失败：{exc}\n')


if __name__ == '__main__':
    main()
