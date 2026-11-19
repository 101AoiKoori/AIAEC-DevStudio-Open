"""Render target-present counterparts to the same train-only colored props."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys

import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from datasets.scripts.fire.render_hard_negative_preview import build_scene,set_variant,write_json,clone_objects
from datasets.scripts.fire.render_blender_fire_dataset import material_override,render_id_pass
from datasets.scripts.fire.fire_blender_labels import visible_bbox,yolo_row
from datasets.scripts.fire.render_hard_negatives import KINDS
from lab.training_common.production_guard import generation_guard

CONFIG=ROOT/'datasets/scripts/fire/config/training_preparation.json'


def render():
    cfg=json.loads(CONFIG.read_text(encoding='utf8'));recipe=json.loads((ROOT/cfg['preview_config']).read_text(encoding='utf8'))
    snapshot=ROOT/cfg['source_snapshot'];output=ROOT/cfg['render_output']
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=snapshot.resolve():raise ValueError('Open the frozen mother snapshot')
    if (output/'manifest.json').exists():raise ValueError('Completed contrast stage already exists')
    recipe['scene_name']='Fire Positive Contrast Batch';recipe['samples']=cfg['samples']
    states=[]
    for category,budget in cfg['train_types'].items():
        for index in range(budget):
            count=1 if index%2==0 else 3
            params={'category':category,'count':count,'index':index,'seed':cfg['seed']+len(states)*1009}
            states.append({'id':f'contrast_positive_{len(states)+1:06d}','failure_event':'color_context_shortcut',
                'min_count':1,'max_count':1,'prior':{'alpha':1,'beta':1},'exposure_weight':1,'failure_cost':1,
                'conditions':[category,f'count:{count}'],'background':{'scene':recipe['scene_name']},
                'target':{'count':count,'building':'A','distractor':category},'labels':{'class_id':0,'scope':'entire_output'},
                'augmentation':recipe['appearance'],'sources':[cfg['source_snapshot']],'parameters':params})
    plan={'schema_version':1,'model_key':'fire_sticker','target_split':'train','purpose':cfg['purpose'],
        'budget':len(states),'uncertainty_weight':.5,'independence_basis':'Same train mother and props as hard negatives',
        'constraints':{'supervision':'all_visible_designated_stickers'},'strata':states}
    with generation_guard('fire.contrast_positive_render',[snapshot],[output],{'config':cfg},
        procedural_inputs=[CONFIG,ROOT/cfg['preview_config'],CONFIG.with_name('source.yaml'),CONFIG.with_name('dataset.yaml')],
        code_paths=[Path(__file__),ROOT/'datasets/scripts/fire/render_hard_negative_preview.py',ROOT/'datasets/scripts/fire/render_blender_fire_dataset.py',ROOT/'datasets/scripts/fire/fire_blender_labels.py'],
        production_plan=plan,sample_output_paths=[output/'images'],auxiliary_output_paths=[output/'instance_pass']) as run:
        if run.reused:return
        rig=build_scene(recipe);scene=rig['scene'];paper=rig['paper'];mask=material_override()
        extras=[]
        for index in range(2):
            obj=paper.copy();obj.data=paper.data.copy();obj.name=f'Fire contrast sticker {index+2}'
            scene.collection.objects.link(obj);extras.append(obj)
        for part in ['images','labels','instance_pass']:(output/part).mkdir(parents=True,exist_ok=True)
        records=[];digest=hashlib.sha256(snapshot.read_bytes()).hexdigest()
        for state in states:
            p=state['parameters'];kind,kind_index=KINDS[p['category']];current=deepcopy(recipe);current['seed']=p['seed']
            small=p['index']%8 in (0,1)
            current['sticker_height_fraction']=cfg['small_sticker_height_fraction' if small else 'normal_sticker_height_fraction']
            color='red' if kind in ('signal_lamp','bin') else 'orange'
            if kind=='bin' and p['index']%3==2:color='orange'
            actual=set_variant(rig,current,kind,color,kind_index)
            paper.hide_render=False;rng=random.Random(p['seed']+77)
            # Disjoint facade height slots keep all three papers on the same wall.
            visible=[paper]
            for index,obj in enumerate(extras):
                obj.hide_render=p['count']==1
                if p['count']==3:visible.append(obj)
            height_centers=[.13,.22,.31] if p['count']==3 else [rng.choice([.13,.22,.31])]
            for obj,z in zip(visible,height_centers):
                size=.4*rng.uniform(*current['sticker_height_fraction']);center=Vector((rng.uniform(-.055,.055),-.0632,z))
                inverse=obj.matrix_world.inverted()
                for vertex,original in zip(obj.data.vertices,rig['paper_vertices']):
                    world=paper.matrix_world@original;delta=world-rig['paper_center']
                    vertex.co=inverse@(center+Vector((delta.x*size/rig['paper_height'],0,delta.z*size/rig['paper_height'])))
                obj.data.update()
            bpy.context.view_layer.update()
            image=output/'images'/(state['id']+'.png');label=output/'labels'/(state['id']+'.txt')
            scene.render.filepath=str(image);bpy.ops.render.render(write_still=True)
            targets=[{'object':obj,'id':i+1} for i,obj in enumerate(visible)]
            props=[{'object':obj,'id':i+4} for i,obj in enumerate(rig['props'][kind])]
            ids=render_id_pass(scene,targets+props,output/'instance_pass'/image.name,mask)
            instances=[]
            for target in targets:
                bbox,area=visible_bbox(ids==target['id'])
                if bbox is None or area<40:raise ValueError('A designated contrast sticker is not visible')
                error=max(abs((target['object'].matrix_world@v.co).y+.0632) for v in target['object'].data.vertices)
                instances.append({'id':target['id'],'xyxy':bbox,'visible_pixels':area,'building':'A','origin':'rendered',
                    'wall_contact':{'max_plane_error_m':error,'offset_m':.0001}})
            box,area=visible_bbox(ids>=4)
            if box is None or area<100:raise ValueError('Contrast prop is not visible')
            label.write_text(''.join(yolo_row(i['xyxy'],*recipe['image_size'])+'\n' for i in instances),encoding='utf8')
            row={'image':f'images/{image.name}','label':f'labels/{label.name}','instance_pass':f'instance_pass/{image.name}',
                'branch':'blender_render','generation_method':'blender_render','split':'train','source_split':'train',
                'source_group':cfg['source_scene'],'parents':[cfg['source_scene']],'source_scene':cfg['source_scene']+'#'+recipe['scene_name'],
                'source_blend_sha256':digest,'scene_snapshot':cfg['source_snapshot'],'instances':instances,
                'building_counts':{'A':{'visible_count':len(instances),'whole_building_count':None,'scope':'visible_output'}},
                'production_state_id':state['id'],'sample_seed':p['seed'],'contrast_positive':True,
                'context':{'distractor':p['category']},'distractor':{'kind':kind,'color':color,'hard_negative_type':p['category'],
                    'xyxy':box,'visible_pixels':area,'object_ids':[v['id'] for v in props]},'actual':actual,
                'size_control':'small' if small else 'normal','supervision_scope':'entire_output_frame'}
            records.append(row);run.record_sample(state['id'],output=image,sources=state['sources'],parameters=p,actual=row)
            write_json(output/'progress.json',{'completed':len(records),'total':len(states)})
        write_json(output/'manifest.json',{'task':'fire','status':'complete','generation_plan':run.plan,'records':records,'images':len(records)})
        run.set_result({'images':len(records),'instances':sum(len(r['instances']) for r in records)})


if __name__=='__main__':render()
