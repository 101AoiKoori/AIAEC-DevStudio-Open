"""Materialize visually annotated bin photos and deterministic foreground masks."""
from hashlib import sha256
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'training_packages/waste_bin_state/Reshoot'
RAW = ROOT / 'datasets/raw/waste_bin_state/reshoot'
OUT = ROOT / 'datasets/annotated/waste_bin_state/reshoot'
# Visually traced at 576x1024. Include the attached lid, rim and handles.
CONTOURS = {
    '173316': [(106,243),(123,234),(129,211),(139,199),(187,190),(192,181),(410,176),(420,188),(454,191),(469,202),(471,226),(489,233),(494,247),(486,271),(480,290),(465,308),(455,313),(424,703),(420,716),(405,721),(180,715),(168,708),(136,296),(123,296),(111,281)],
    '173321': [(50,248),(60,237),(80,240),(96,231),(130,229),(131,218),(143,204),(190,197),(192,186),(400,187),(413,200),(459,205),(473,214),(473,236),(493,243),(501,254),(491,283),(476,302),(466,307),(433,705),(421,713),(186,712),(173,705),(140,317),(128,319),(116,308),(91,311),(83,301),(86,286),(62,275)],
    '173327': [(67,222),(87,213),(92,186),(109,176),(129,173),(130,151),(141,146),(181,144),(196,152),(332,152),(338,143),(380,146),(388,154),(402,151),(407,170),(441,174),(453,186),(454,207),(473,218),(479,239),(470,271),(457,288),(446,294),(410,713),(398,725),(160,725),(146,718),(98,283),(82,283),(75,264)],
    '173331': [(0,155),(23,147),(27,125),(41,108),(104,101),(117,89),(386,87),(401,99),(450,104),(474,116),(480,150),(506,159),(542,155),(560,156),(575,170),(571,189),(554,209),(537,220),(526,248),(492,266),(471,275),(418,755),(403,769),(101,770),(89,760),(36,238),(13,230),(0,214)],
    '173354': [(86,336),(104,325),(116,297),(137,279),(197,273),(200,254),(229,252),(232,272),(289,272),(292,252),(328,252),(331,274),(384,279),(405,295),(421,321),(441,332),(447,347),(440,374),(419,382),(373,680),(363,691),(192,693),(179,687),(113,382),(92,376)],
    '173400': [(32,313),(40,295),(68,293),(88,278),(89,268),(111,268),(138,284),(178,270),(439,268),(461,277),(482,303),(486,320),(498,330),(503,349),(498,368),(476,379),(421,701),(409,713),(208,713),(194,704),(169,590),(144,618),(111,667),(96,680),(80,680),(67,666),(32,449),(30,414),(37,393),(68,368),(64,347),(41,334)],
    '173404': [(72,305),(94,280),(120,262),(147,248),(365,247),(398,258),(426,280),(457,290),(475,307),(477,331),(460,345),(440,349),(432,390),(454,402),(475,424),(486,449),(450,698),(436,718),(419,731),(367,738),(345,746),(249,746),(241,740),(134,736),(117,725),(102,708),(73,470),(72,444),(82,422),(99,405),(124,394),(111,347),(81,340)],
    '173411': [(49,299),(69,286),(91,267),(119,252),(390,248),(424,249),(445,233),(462,237),(479,250),(489,257),(494,278),(486,302),(462,314),(461,334),(485,345),(498,371),(496,429),(475,622),(463,642),(446,642),(426,632),(414,607),(394,489),(367,650),(354,665),(173,666),(159,660),(89,338),(63,332),(53,323)],
}
HOLES = {'173400': [[(148,455),(170,576),(148,610)]],
         '173411': [[(401,371),(396,408),(390,458),(380,528),(369,606),(365,646),
                     (422,641),(412,610),(409,580),(410,530),(412,480),(412,430),(409,401)]]}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    for directory in (RAW, OUT, OUT/'labels', OUT/'masks', OUT/'cutouts', OUT/'overlays'):
        directory.mkdir(parents=True, exist_ok=True)
    specifications = [{'suffix':suffix, 'polygon':polygon, 'color':'gray',
        'state':'closed' if i<4 else 'open', 'holes':HOLES.get(suffix,[]),
        'composite_eligible':suffix!='173331', 'truncated':suffix=='173331'}
        for i,(suffix,polygon) in enumerate(CONTOURS.items())]
    for name in ('reshoot_red_blue_annotations.json','reshoot_green_annotations.json',
                 'reshoot_gray_extra_annotations.json'):
        specifications.extend(json.loads((Path(__file__).parent/'config'/name).read_text(encoding='utf8'))['records'])
    declared={f'IMG_20261118_{item["suffix"]}.jpg' for item in specifications}
    if declared != {p.name for p in SOURCE.glob('*.jpg')} or len(declared)!=len(specifications):
        raise ValueError('Every supplied photo needs exactly one visual annotation')
    records = []
    for index, specification in enumerate(specifications):
        suffix,polygon=specification['suffix'],specification['polygon']
        color,state=specification['color'],specification['state']
        holes=specification.get('holes',[])
        source = SOURCE / f'IMG_20261118_{suffix}.jpg'
        image = ImageOps.exif_transpose(Image.open(source)).convert('RGB')
        if image.size != (2304,4096):
            raise ValueError('Annotation coordinates require 2304x4096 upright photos')
        target = RAW / source.name
        shutil.copy2(source, target)
        small = np.array(image.resize((576,1024)))
        outline = np.zeros((1024,576),np.uint8)
        cv2.fillPoly(outline,[np.array(polygon,np.int32)],1)
        for hole in holes:
            cv2.fillPoly(outline,[np.array(hole,np.int32)],0)
        mask = np.full(outline.shape,cv2.GC_PR_BGD,np.uint8)
        mask[outline>0] = cv2.GC_PR_FGD
        mask[cv2.erode(outline,np.ones((13,13),np.uint8))>0] = cv2.GC_FGD
        mask[cv2.dilate(outline,np.ones((11,11),np.uint8))==0] = cv2.GC_BGD
        for hole in holes:
            cv2.fillPoly(mask,[np.array(hole,np.int32)],cv2.GC_BGD)
        cv2.setRNGSeed(42+index)
        cv2.grabCut(cv2.cvtColor(small,cv2.COLOR_RGB2BGR),mask,None,
                    np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),4,cv2.GC_INIT_WITH_MASK)
        foreground = np.isin(mask,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)*255
        full = cv2.resize(foreground,image.size,interpolation=cv2.INTER_NEAREST)
        mask_path = OUT/'masks'/f'{source.stem}.png'
        Image.fromarray(full).save(mask_path)
        # Source boxes follow the visually traced whole-bin envelope, independently of mask fitting.
        points = np.array(polygon)*4
        x0,y0 = points.min(axis=0); x1,y1 = np.minimum(points.max(axis=0)+4,image.size)
        box = [int(x0),int(y0),int(x1),int(y1)]
        ys,xs = np.nonzero(full)
        crop = [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
        rgba=image.convert('RGBA');rgba.putalpha(Image.fromarray(full))
        cutout=OUT/'cutouts'/f'{source.stem}.png';rgba.crop(crop).save(cutout)
        cid=1 if state=='closed' else 0
        label=OUT/'labels'/f'{source.stem}.txt'
        label.write_text(f'{cid} {(x0+x1)/2/2304:.9f} {(y0+y1)/2/4096:.9f} {(x1-x0)/2304:.9f} {(y1-y0)/4096:.9f}\n',encoding='utf8')
        preview=image.copy();preview.thumbnail((576,1024));draw=ImageDraw.Draw(preview)
        draw.rectangle(tuple(v/4 for v in box),outline='lime',width=3)
        draw.text((10,10),f'{color} {state} / whole bin',fill='lime');preview.save(OUT/'overlays'/source.name)
        records.append({'id':source.stem,'source_record_id':source.stem,'image':relative(target),
            'source_image':relative(source),'label':relative(label),'mask':relative(mask_path),
            'cutout':relative(cutout),'cutout_xyxy':crop,'image_size':list(image.size),
            'bins':[{'color':color,'state':state,'bbox_xyxy':box}], 'color':color,'state':state,
            'split':'train','capture_session':'bins_reshoot_20261118',
            'physical_parent_group':'bins_reshoot_20261118','source_sha256':sha256(source.read_bytes()).hexdigest(),
            'mask_sha256':sha256(mask_path.read_bytes()).hexdigest(),
            'composite_eligible':specification.get('composite_eligible',True),
            'truncated':specification.get('truncated',False),
            'annotation':{'method':'codex_visual_whole_image_box_and_polygon_review',
                          'completeness':'complete','human_confirmed':False,
                          'state_evidence':'lid seated over rim' if state=='closed' else 'open mouth and rear-folded attached lid',
                          'mask_method':'visually seeded GrabCut; silhouette QA required'}})
    document={'schema_version':1,'task':'waste_bin_state','class_names':['open','closed'],
              'status':'visually_annotated_sources','capture_session':'bins_reshoot_20261118',
              'records':records,'scope':'Four physical bins in one indoor capture session; per-color/state counts follow actual records; training only.'}
    (OUT/'manifest.json').write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'images':len(records),'manifest':relative(OUT/'manifest.json')}))


if __name__=='__main__':
    main()
