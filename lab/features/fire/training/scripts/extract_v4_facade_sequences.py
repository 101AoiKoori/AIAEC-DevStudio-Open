"""Decode the two existing facade videos into traceable 640x360 frame sequences."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from lab.training_common.production_guard import generation_guard
OUT=ROOT/'datasets/annotated/fire/v4_facade_video_sequences_20261112'
VIDEOS=(('scene_001','Archive/Assets/working_sources/legacy/site/场景/三楼与场地视频/三栋楼正向平移_垃圾桶入画_001.mp4'),
        ('scene_008','Archive/Assets/working_sources/legacy/site/场景/三楼与场地视频/三栋楼正向横向扫描_008.mp4'))


def main():
    inputs=[ROOT/relative for _,relative in VIDEOS]
    with generation_guard('fire.extract_v4_facade_sequences',inputs,[OUT],
                          {'videos':list(VIDEOS),'resolution':[640,360]},
                          code_paths=[Path(__file__)],root=ROOT) as run:
        if run.reused:
            print(json.dumps({'status':'reused','output':OUT.relative_to(ROOT).as_posix()}))
            return
        _extract(run.source_refs)


def _extract(source_refs):
    OUT.mkdir(parents=True,exist_ok=False)
    metadata={'status':'running','resolution':[640,360],'decoder':'ffmpeg area resize and showinfo PTS',
              'scope':'provided existing videos; physical scene and assets may recur; no independence inferred',
              'videos':[]}
    try:
        for sid,relative in VIDEOS:
            source=(ROOT/relative).resolve()
            if not source.is_relative_to(ROOT) or not source.is_file():raise ValueError('missing workspace video')
            target=OUT/sid;target.mkdir();log=target/'ffmpeg.log'
            command=['ffmpeg','-nostdin','-hide_banner','-loglevel','info','-threads','4','-i',str(source),
                     '-vf','scale=640:360:flags=area,showinfo','-fps_mode','passthrough',
                     str(target/'frame_%06d.png')]
            with log.open('wb') as stream:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                                      creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=300)
            if result.returncode:raise RuntimeError('ffmpeg failed for '+sid)
            points=[]
            for line in log.read_text(encoding='utf8',errors='replace').splitlines():
                m=re.search(r'\bn:\s*(\d+)\s+pts:\s*-?\d+\s+pts_time:([\d.]+)',line)
                if m:points.append({'frame_number':int(m.group(1))+1,'time_seconds':float(m.group(2))})
            count=len(list(target.glob('frame_*.png')))
            if count==0 or count!=len(points):raise ValueError('frame and PTS count mismatch for '+sid)
            (target/'timestamps.json').write_text(json.dumps(points,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            metadata['videos'].append({'id':sid,'source_video':relative,'frames':count,'fps_status':'use decoded PTS',
                'timestamp_index':(target/'timestamps.json').relative_to(ROOT).as_posix(),
                'decode_log':log.relative_to(ROOT).as_posix(),
                'source_references':source_refs[relative]})
        metadata['status']='complete'
    except BaseException as error:
        metadata['status']='failed_or_interrupted';metadata['error']=str(error);raise
    finally:
        (OUT/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(metadata,ensure_ascii=False))


if __name__=='__main__':main()
