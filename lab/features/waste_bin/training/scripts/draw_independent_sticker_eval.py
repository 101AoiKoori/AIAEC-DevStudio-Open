"""Draw new symbolic waste illustrations for split-specific visual review.

The drawings are generated from geometric primitives, without loading the 24
training icon images. They depict broad concepts and are not competition
artwork. The same drawing program is used for both variants; evaluation must
record that shared procedural origin and cannot claim real-photo accuracy.
"""
import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[5]
CANDIDATE_CLASSES = (1, 3, 4, 5, 7, 8, 10, 11, 12, 15, 18, 23)


def draw_test_independent(index):
    """Draw structurally distinct test illustrations for a limited subset."""
    im=Image.new('RGB',(160,160),'white');d=ImageDraw.Draw(im);ink='#25332d';g='#62864c';brown='#ad7b54'
    if index==1:
        d.polygon([(45,46),(107,29),(124,116),(61,132)],fill='#c6cac7',outline=ink,width=6)
        d.ellipse((43,36,107,57),fill='#e8e8dd',outline=ink,width=5);d.rectangle((58,74,113,99),fill='#b65e52')
    elif index==3:
        d.polygon([(29,43),(114,43),(132,103),(47,112)],fill='#ddd9cc',outline=ink,width=5)
        d.ellipse((26,43,56,112),fill='#c7c4b8',outline=ink,width=5)
        for y in (60,72,84,96):d.line((58,y,117,y-2),fill=ink,width=3)
    elif index==4:
        d.polygon([(35,49),(98,34),(130,48),(66,64)],fill='#d2a374',outline=ink,width=5)
        d.polygon([(35,49),(66,64),(66,128),(35,109)],fill='#b27f55',outline=ink,width=5)
        d.polygon([(66,64),(130,48),(130,111),(66,128)],fill='#ce9e6d',outline=ink,width=5)
        d.line((80,62,80,121),fill='#f0d297',width=8)
    elif index==5:
        d.polygon([(45,37),(66,25),(94,25),(114,37),(129,56),(115,70),(104,60),(104,129),(56,129),(56,60),(45,70),(31,56)],fill='#c0c7c6',outline=ink,width=6)
        d.polygon([(66,25),(80,43),(94,25)],fill='white',outline=ink,width=3);d.rectangle((84,67,98,82),outline=ink,width=3)
    elif index==7:
        d.polygon([(28,115),(35,83),(49,57),(76,32),(106,27),(125,47),(129,74),(108,105),(72,123)],fill=g,outline=ink,width=5)
        d.line((38,120,105,46),fill=ink,width=5);d.line((66,88,51,68),fill=ink,width=3);d.line((81,72,105,76),fill=ink,width=3)
    elif index==8:
        d.polygon([(55,42),(104,42),(96,64),(91,80),(98,105),(105,124),(54,124),(62,104),(69,82),(64,64)],fill='#e9d8aa',outline=ink,width=5)
        d.line((58,42,102,42),fill='#aa5643',width=9);d.line((57,124,103,124),fill='#aa5643',width=9);d.line((80,40,68,22),fill=ink,width=5)
        d.ellipse((71,70,77,79),fill=ink);d.ellipse((84,84,90,93),fill=ink)
    elif index==10:
        d.rounded_rectangle((30,49,110,127),radius=16,fill=brown,outline=ink,width=5)
        d.rounded_rectangle((66,38,128,120),radius=12,fill='#e9c48c',outline=ink,width=5)
        d.line((83,51,113,51),fill='#b68d63',width=4)
    elif index==11:
        d.polygon([(23,82),(43,67),(51,82),(43,97)],fill='#dfddd3',outline=ink,width=4);d.ellipse((30,76,35,81),fill=ink)
        d.line((48,82,124,82),fill=ink,width=6)
        for x in (63,78,93,108):d.line((x,82,x-10,63),fill=ink,width=4);d.line((x,82,x-10,102),fill=ink,width=4)
        d.polygon([(124,82),(143,65),(143,99)],fill='#dfddd3',outline=ink,width=4)
    elif index==12:
        d.ellipse((39,29,121,111),fill='#d6d9d6',outline=ink,width=7);d.ellipse((48,38,112,102),outline='#8b9690',width=4)
        d.line((66,69,94,69),fill=ink,width=5);d.line((80,55,80,83),fill=ink,width=5)
        d.arc((41,40,120,121),0,135,fill=ink,width=6)
    elif index==15:
        d.rounded_rectangle((28,50,85,76),radius=13,fill='#e2b796',outline=ink,width=5);d.line((57,50,57,76),fill=ink,width=4)
        d.rounded_rectangle((75,86,132,112),radius=13,fill='#a9cbd0',outline=ink,width=5);d.line((104,86,104,112),fill=ink,width=4)
        d.ellipse((38,96,63,121),fill='#e5e1db',outline=ink,width=4)
    elif index==18:
        d.rounded_rectangle((34,67,120,94),radius=5,fill='#ecebe2',outline=ink,width=5)
        d.rectangle((90,69,120,92),fill='#c79a63');d.line((34,67,34,94),fill='#a34f37',width=7)
        d.line((30,59,26,50),fill='#858b84',width=3)
    elif index==23:
        d.rounded_rectangle((17,53,69,96),radius=8,outline=ink,width=6);d.rounded_rectangle((91,53,143,96),radius=8,outline=ink,width=6)
        d.line((69,65,91,65),fill=ink,width=6);d.line((17,58,8,44),fill=ink,width=5);d.line((143,58,152,44),fill=ink,width=5)
    else:
        raise ValueError('test artwork not approved for this class')
    return im


def draw_icon(index, variant):
    image = Image.new('RGB', (160, 160), 'white')
    d = ImageDraw.Draw(image)
    ink = '#30404a' if variant == 'val' else '#172432'
    accent = '#629fa3' if variant == 'val' else '#9a7766'
    green = '#558b5a' if variant == 'val' else '#607e38'
    gray = '#b8c2c4' if variant == 'val' else '#a3acb1'
    tan = '#d3ad76' if variant == 'val' else '#ce9659'
    width = 5 if variant == 'val' else 7
    if index == 0:  # plastic bottle
        d.polygon([(63,22),(97,22),(97,42),(108,55),(108,131),(52,131),(52,55),(63,42)],fill='#c9ebef',outline=ink,width=width)
        d.rectangle((64,14,96,23),fill=accent); d.line((57,82,103,82),fill=accent,width=5)
    elif index == 1:  # metal drink can
        d.rounded_rectangle((50,28,110,135),radius=12,fill=gray,outline=ink,width=width)
        d.ellipse((50,22,110,43),fill='#dce1df',outline=ink,width=width);d.arc((59,29,100,37),0,180,fill=ink,width=3)
    elif index == 2:  # glass bottle
        d.polygon([(68,20),(93,20),(93,47),(105,67),(105,132),(55,132),(55,67),(68,47)],fill='#b1dbc4',outline=ink,width=width)
        d.rectangle((67,13,94,21),fill=green);d.line((62,88,99,88),fill=green,width=4)
    elif index == 3:  # newspaper
        d.polygon([(28,39),(126,30),(133,117),(36,127)],fill='#d3d5cf',outline=ink,width=width)
        d.rectangle((42,52,66,76),fill=accent)
        for y in (53,65,77,90,101,112):d.line((73,y,117,y-4),fill=ink,width=3)
    elif index == 4:  # open carton
        d.polygon([(31,62),(80,44),(129,62),(80,80)],fill=tan,outline=ink,width=width)
        d.polygon([(31,62),(80,80),(80,132),(31,113)],fill='#bc8654',outline=ink,width=width)
        d.polygon([(80,80),(129,62),(129,113),(80,132)],fill='#d2a06d',outline=ink,width=width)
        d.polygon([(30,61),(15,39),(65,27),(80,44)],fill=tan,outline=ink,width=width)
        d.polygon([(80,44),(98,25),(145,40),(129,62)],fill=tan,outline=ink,width=width)
    elif index == 5:  # old T-shirt
        d.polygon([(54,28),(69,21),(91,21),(106,28),(132,44),(118,70),(105,64),(105,132),(55,132),(55,64),(42,70),(28,44)],fill=gray,outline=ink,width=width)
        d.arc((68,16,92,43),0,180,fill=ink,width=4);d.line((69,90,90,101),fill=accent,width=3)
    elif index == 6:  # leftover rice
        d.ellipse((28,88,132,132),fill='#dbd0b5',outline=ink,width=width)
        d.pieslice((36,40,124,125),180,360,fill='#f0e6ca',outline=ink,width=4)
        rng=random.Random(620+int(variant=='test'))
        for _ in range(35):
            x=rng.randint(50,110);y=rng.randint(72,98);d.line((x,y,x+4,y-3),fill='#9b8562',width=2)
    elif index == 7:  # vegetable leaves
        for box in ((29,29,85,100),(73,27,127,96),(55,57,111,128)):
            d.ellipse(box,fill=green,outline=ink,width=4)
        d.line((80,130,80,51),fill=ink,width=4)
    elif index == 8:  # apple core
        d.polygon([(55,35),(105,35),(96,58),(100,81),(108,122),(52,122),(60,81),(64,58)],fill='#f1daa2',outline=ink,width=width)
        d.arc((47,21,112,57),180,360,fill='#b45d4a',width=10);d.arc((47,105,112,139),0,180,fill='#b45d4a',width=10)
        d.line((80,35,89,16),fill=ink,width=5);d.ellipse((73,67,78,75),fill=ink);d.ellipse((88,75,93,83),fill=ink)
    elif index == 9:  # tea leaves
        rng=random.Random(99+int(variant=='test'))
        for _ in range(25):
            x=rng.randint(37,115);y=rng.randint(53,115);d.line((x,y,x+rng.randint(-8,8),y-rng.randint(5,14)),fill='#584a31',width=5)
    elif index == 10:  # bread
        d.rounded_rectangle((42,36,118,130),radius=18,fill=tan,outline='#96643f',width=8)
        d.arc((39,22,122,73),180,360,fill='#96643f',width=8)
    elif index == 11:  # fish bone
        d.line((35,80,119,80),fill=ink,width=6)
        d.ellipse((20,68,44,92),outline=ink,width=5);d.polygon([(117,80),(141,62),(141,99)],outline=ink,width=5)
        for x in (55,70,85,100):d.line((x,80,x+10,62),fill=ink,width=4);d.line((x,80,x+10,98),fill=ink,width=4)
    elif index == 12:  # button cell
        d.ellipse((34,47,126,113),fill=gray,outline=ink,width=width);d.ellipse((42,38,118,101),fill='#dce2e0',outline=ink,width=width)
        d.line((65,68,95,68),fill=ink,width=4);d.line((80,53,80,83),fill=ink,width=4)
    elif index == 13:  # fluorescent tube
        d.rounded_rectangle((69,17,91,102),radius=10,fill='#edeeee',outline=ink,width=5)
        d.rounded_rectangle((64,99,96,125),radius=5,fill=gray,outline=ink,width=4)
        d.line((69,111,91,111),fill=ink,width=3);d.line((69,118,91,118),fill=ink,width=3)
    elif index == 14:  # mercury thermometer
        d.rounded_rectangle((70,19,90,113),radius=9,fill='#e9eeee',outline=ink,width=5)
        d.ellipse((64,104,96,136),fill='#d8e5e5',outline=ink,width=5);d.line((80,40,80,119),fill='#c04a3a',width=5)
        for y in range(47,100,10):d.line((84,y,91,y),fill=ink,width=2)
    elif index == 15:  # pills
        d.rounded_rectangle((25,47,94,70),radius=12,fill='#e8c58e',outline=ink,width=4);d.line((59,48,59,69),fill=ink,width=4)
        d.ellipse((77,77,117,117),fill='#e8d6cf',outline=ink,width=4);d.line((87,97,106,97),fill=ink,width=4)
        d.ellipse((34,92,68,126),fill='#f0eee9',outline=ink,width=4)
    elif index == 16:  # paint can
        d.rectangle((44,54,116,127),fill=gray,outline=ink,width=5);d.ellipse((44,43,116,67),fill=green,outline=ink,width=5)
        d.line((51,68,51,89,61,93,61,72),fill=green,width=8);d.arc((54,19,106,74),180,360,fill=ink,width=4)
    elif index == 17:  # film roll
        d.rounded_rectangle((51,38,105,127),radius=8,fill='#333a3c',outline=ink,width=5)
        d.ellipse((50,29,106,47),fill=gray,outline=ink,width=4);d.rectangle((102,54,121,105),fill='#8b7054',outline=ink,width=4)
        for y in (60,73,86,99):d.rectangle((109,y,115,y+5),fill='#dfd6b8')
    elif index == 18:  # cigarette butt
        d.rounded_rectangle((27,70,130,92),radius=8,fill='#f3f2ea',outline=ink,width=4)
        d.rectangle((98,72,130,90),fill=tan);d.line((27,70,27,92),fill='#9c4b36',width=6)
    elif index == 19:  # used tissue
        d.polygon([(32,60),(53,34),(75,45),(92,29),(126,55),(114,82),(128,105),(102,126),(76,115),(56,129),(29,104),(39,82)],fill='#e8eceb',outline=ink,width=4)
        d.line((53,55,77,79,101,52),fill=gray,width=4);d.line((50,98,76,79,105,105),fill=gray,width=4)
    elif index == 20:  # disposable chopsticks
        d.polygon([(57,25),(65,24),(72,133),(64,134)],fill=tan,outline=ink,width=3)
        d.polygon([(86,25),(94,26),(88,134),(80,133)],fill=tan,outline=ink,width=3)
    elif index == 21:  # broken pottery
        d.polygon([(40,35),(109,30),(104,74),(91,80),(82,67),(71,87),(58,77),(44,84)],fill='#ede6d8',outline=ink,width=5)
        d.polygon([(45,100),(70,88),(79,119),(52,130)],fill=gray,outline=ink,width=4)
        d.polygon([(91,95),(116,87),(123,115),(100,128)],fill='#ede6d8',outline=ink,width=4)
    elif index == 22:  # shell
        d.pieslice((31,28,129,130),180,360,fill='#e4cfaa',outline=ink,width=5)
        d.polygon([(31,79),(44,121),(80,136),(116,121),(129,79)],fill='#e4cfaa',outline=ink,width=5)
        for x in (40,54,67,80,93,106,120):d.line((80,131,x,80),fill='#a58061',width=3)
    elif index == 23:  # glasses
        d.ellipse((20,51,72,100),outline=ink,width=6);d.ellipse((88,51,140,100),outline=ink,width=6)
        d.arc((69,58,91,80),180,360,fill=ink,width=5);d.line((20,60,11,45),fill=ink,width=5);d.line((140,60,149,45),fill=ink,width=5)
    else:
        raise ValueError('invalid class')
    return image


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--distinct-subset',action='store_true');args=p.parse_args()
    out=(ROOT/args.out).resolve()
    if not out.is_relative_to((ROOT/'datasets/exports/waste_bin').resolve()) or out.exists():
        p.error('output must be new and under datasets/exports/waste_bin')
    task=next(t for t in json.loads((ROOT/'config/perception_tasks.json').read_text(encoding='utf8'))['tasks'] if t['id']=='waste_bin')
    names=list(task['waste_items'])
    if len(names)!=24: raise ValueError('expected 24 classes')
    out.mkdir(parents=True)
    font=ImageFont.truetype('msyh.ttc',18)
    manifest={'status':'preview_only','source':'new procedural vector drawings; no training icon files read',
              'shared_generator':'both sets share the same drawing program; distinct subset uses separate test drawing code', 'sets':{}}
    for variant in ('val','test'):
        (out/variant).mkdir();board=Image.new('RGB',(1280,960),'#dddddd');d=ImageDraw.Draw(board)
        ids=CANDIDATE_CLASSES if args.distinct_subset else range(24)
        for cell,i in enumerate(ids):
            name=names[i];icon=draw_test_independent(i) if args.distinct_subset and variant=='test' else draw_icon(i,variant)
            icon.save(out/variant/f'{i:02d}.png')
            x=(cell%8)*160;y=(cell//8)*320;board.paste(icon,(x,y+42));d.text((x+5,y+10),f'{i} {name}',font=font,fill='black')
        board.save(out/f'{variant}_gallery.png')
        manifest['sets'][variant]=[{'class_id':i,'name':names[i],'artwork':f'{variant}/{i:02d}.png'} for i in ids]
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('preview artwork:',len(CANDIDATE_CLASSES) if args.distinct_subset else len(names),'classes x 2')


if __name__=='__main__':main()
