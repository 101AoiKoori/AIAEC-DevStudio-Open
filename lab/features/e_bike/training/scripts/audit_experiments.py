"""Inventory e_bike research artifacts and diagnose the recorded 17/26 reference.

Reads existing files only; writes one evidence index. Never moves or deletes runs.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
import yaml
from lab.training_common.engine.reporting import match_detections
from lab.features.e_bike.training.scripts.report_pose_counts import pose_counts

REFERENCE='runs/e_bike/keypoint_control_pipeline_20261118/evaluation/box_03/predictions.jsonl'
ROLES={
    'p2_detector_20261118':'limited_update_8image_fit_before_confidence_convergence',
    'p2_detector_densefit_20261118':'8image_repeated_fit_and156train_equal_update_P2_P3_control',
    'pose_aware_probe_20261118':'superseded_row_match_metadata_metrics_retained',
    'pose_aware_checked_20261118':'three_seed_posture_partition_nms_no_net_gain',
    'research_handoff_20261118':'hash_pinned_semantic20_research_handoff_and_source_reading',

    'quality_rank_20261118':'continuous_iou_and_pairwise_ranking_frozen_features_no_net_gain',
    'verifier_seed_study_20261118':'fixed_crop_training_seed42_43_44_reproducibility',
    'candidate_pool_audit_20261118':'superseded_self_suppressor_annotation_capacity_totals_valid',
    'candidate_pool_checked_20261118':'suppression_conflict_previews_and_iou_matrix',
    'candidate_pool_classified_20261118':'same_target_box_quality_and_multi_gt_merge_diagnostic',
    'matching_ambiguity_audit_20261118':'box_matching_multi_gt_compatibility_sensitivity',

    'source_contact_20261118':'failed_initial_parser_preserved',
    'source_contact_20261118_complete':'immediate_and_v3_inherited_source_pixel_contact_audit',
    'candidate_completeness_20261118':'low_iou_vehicle_candidate_rejection_training_probe',
    'candidate_completeness_control_20261118':'matched_slots_pure_background_repeat_control',
    'completeness_metric_audit_20261118':'strict_metrics_for_partial_reject_arm',
    'completeness_control_metric_audit_20261118':'strict_metrics_for_matched_control',
    'completeness_comparison_20261118':'four_pipeline_comparison_rejection_not_adopted',
    'candidate_latency_20261118':'four_pipeline_offline_compute_timing',
    'candidate_role_split_20261118':'superseded_cached_match_metadata_diagnostic',
    'candidate_role_split_checked_20261118':'semantic_gate_and_posture_head_role_separation_diagnostic',

    'evaluation_audit_20261118':'frozen_prediction_metric_matching_and_family_replay',
    'candidate_verifier_20261118':'frozen_detector_three_class_pre_final_nms_verifier_probe',
    'v4_baseline_20261118':'retained_first_baseline',
    'v4_20261118_083827_863607':'baseline_before_evaluation',
    'synthetic_short_inputs_20261118':'frozen_research_inputs',
    'keypoint_inputs_20261118':'two_point_input_snapshot',
    'four_part_inputs_20261118':'four_part_input_snapshot',
    'synthetic_short_20261118':'short_finetune_not_adopted',
    'synthetic_short_before_20261118':'before_finetune_evaluation',
    'nms_comparison_20261118':'preliminary_mixed_size_batch_diagnostic_superseded',
    'nms_comparison_single_20261118':'rectangular_single_image_nms_comparison',
    'two_stage_20261118':'posture_classifier_and_oracle_comparison',
    'two_stage_detector_20261118':'first_one_class_detector_probe',
    'keypoint_micro_20261118':'two_point_pose_probe',
    'keypoint_control_20261118':'current_research_reference_detector',
    'keypoint_control_pipeline_20261118':'current_research_reference_evaluation',
    'four_part_micro_20261118':'four_part_pose_training',
    'four_part_best_20261118':'four_part_best_checkpoint_evaluation',
    'four_part_control_final_20261118':'matched_detector_final_checkpoint_evaluation',
    'rtdetr_micro_20261118':'rtdetr_initial_bias_warmup_diagnostic',
    'rtdetr_biasfix_20261118':'rtdetr_corrected_bias_warmup_probe',
    'point_counter_micro_20261118':'independent_point_probe_training_fit_insufficient',
    'part_hull_probe_20261118':'fixed_part_ownership_hull_postprocessing_probe',
    'input_scale_probe_20261118':'frozen_input_scale_and_exact_pre_nms_diagnostic',
    'threshold_nms_probe_20261118':'confidence_hard_soft_nms_replay_and_combined_review',
    'threshold_refine_20261118':'bounded40_epoch_detector_refinement_not_adopted',
    'threshold_refine_last_eval_20261118':'refinement_final_checkpoint_four_setting_evaluation',
    'threshold_refine_best_eval_20261118':'refinement_map_selected_checkpoint_four_setting_evaluation',
    'focused_inputs_20261118':'focused_synthetic_real_crop_and_hard_negative_research_inputs',
    'focused_train_20261118':'focused_data_bounded40_epoch_detector_training',
    'focused_comparison_20261118':'original_final_best_on_old26_and_new32_layout_holdout',
    'data_ablation_20261118':'equal_slot_four_arm_data_ablation_and_pose_localization_diagnosis',
    'real_sampling_followup_20261118':'paired_original_real_parent_resampling_followup_and_combined_comparison',
    '_controllers':'execution_controller_records',
    'runs':'nested_historical_runs_preserve_in_place',
    'production_20261114':'historical_production_records',
    'authoritative_plans_20261114':'historical_plan_records',
    'safe_legacy_20261114':'historical_legacy_records',
}


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def file_record(p,hash_content=False):
    s=p.stat();result={'path':p.relative_to(ROOT).as_posix(),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
    if hash_content:result['sha256']=digest(p)
    return result


def tensor_role(p):
    if p.parent.name=='features':return 'frozen_feature_cache'
    if p.name=='backbone_state.pt':return 'frozen_backbone_state'
    if p.name=='initialized_pose.pt':return 'pose_initialization'
    return 'model_checkpoint'


def failure_audit():
    rows=[json.loads(l) for l in (ROOT/REFERENCE).read_text(encoding='utf-8').splitlines() if l.strip()]
    counts=pose_counts(rows);failures=[];entity=Counter()
    for row,count in zip(rows,counts['records']):
        gt=row['ground_truth'];pred=row['predictions']
        pairs,miss,extra=match_detections([{**v,'class_id':0} for v in gt],[{**v,'class_id':0} for v in pred],.5)
        wrong=[{'gt_index':i,'prediction_index':j} for i,j,_ in pairs if gt[i]['class_id']!=pred[j]['class_id']]
        entity.update(tp=len(pairs),fp=len(extra),fn=len(miss),matched_posture_errors=len(wrong))
        if not count['both_pose_counts_exact']:
            failures.append({**count,'group':row['group'],'entity_missed_gt_indices':miss,
                             'entity_false_prediction_indices':extra,'matched_posture_errors':wrong,
                             'category':'localization_involved' if miss or extra else 'posture_only',
                             'preview':'runs/e_bike/keypoint_control_pipeline_20261118/evaluation/box_03/'+Path(row['source_image']).stem+'.jpg'})
    return {'source':REFERENCE,'counts':{k:v for k,v in counts.items() if k!='records'},
            'entity_metrics':dict(entity),'failure_categories':dict(Counter(r['category'] for r in failures)),
            'groups':{g:{'images':len([r for r in rows if r['group']==g]),'exact_images':sum(c['both_pose_counts_exact'] for r,c in zip(rows,counts['records']) if r['group']==g)} for g in sorted({r['group'] for r in rows})},'failures':failures}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='datasets/annotated/e_bike/experiment_inventory.json')
    args=p.parse_args();output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT/'datasets/annotated/e_bike'):raise ValueError('Evidence output must be in e_bike annotated directory')
    source=yaml.safe_load((ROOT/'datasets/scripts/e_bike/config/source.yaml').read_text(encoding='utf-8'))
    results=[];references=set();expected_hashes=set();historical_identifiers=set()
    def collect(value,identifier=False):
        if isinstance(value,dict):
            path=value.get('path',value.get('weights'))
            if isinstance(path,str) and path.startswith(('runs/','datasets/','lab/')) and isinstance(value.get('sha256'),str):
                expected_hashes.add((path,value['sha256']))
            for k,v in value.items():
                if isinstance(k,str) and k.startswith('runs/') and isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v):
                    expected_hashes.add((k,v))
            for k,v in value.items():
                collect(k,identifier)
                collect(v,identifier or k in {'declared_parent_groups','historical_parent_label'})
        elif isinstance(value,list):
            for v in value:collect(v,identifier)
        elif isinstance(value,str) and value.startswith(('runs/','datasets/','lab/')) and '\n' not in value:
            (historical_identifiers if identifier else references).add(value)
    for key,value in source.items():
        if key.endswith('_result'):
            path=ROOT/value
            item={'source_key':key,'path':value,'exists':path.is_file()}
            if path.is_file():
                data=json.loads(path.read_text(encoding='utf-8'));collect(data)
                item.update(status=data.get('status'),sha256=digest(path))
            results.append(item)
    runs=[]
    for folder in sorted((ROOT/'runs/e_bike').iterdir()):
        if not folder.is_dir():continue
        files=[f for f in folder.rglob('*') if f.is_file()]
        states=[]
        for f in files:
            if f.name.endswith('status.json') and f.parent.name=='metadata':
                try:
                    s=json.loads(f.read_text(encoding='utf-8'));states.append({'path':f.relative_to(ROOT).as_posix(),'phase':s.get('phase'),'status':s.get('status')})
                except (ValueError,OSError):states.append({'path':f.relative_to(ROOT).as_posix(),'status':'unreadable'})
        runs.append({'path':folder.relative_to(ROOT).as_posix(),'role':ROLES.get(folder.name,'unclassified_preserve_in_place'),
                     'files':len(files),'bytes':sum(f.stat().st_size for f in files),
                     'tensor_artifacts':[{**file_record(f,True),'role':tensor_role(f)} for f in files if f.suffix=='.pt'],
                     'review_pages':[f.relative_to(ROOT).as_posix() for f in files if f.suffix=='.html'],
                     'execution_statuses':states})
    ref_records=[]
    for value in sorted(references):
        f=ROOT/value
        try:exists=f.exists()
        except OSError:exists=False
        ref_records.append({'path':value,'exists':exists})
    known_hashes={p['path']:p['sha256'] for r in runs for p in r['tensor_artifacts']}
    hash_checks=[]
    for path,expected in sorted(expected_hashes):
        actual=known_hashes.get(path)
        if actual is None and (ROOT/path).is_file():actual=digest(ROOT/path)
        hash_checks.append({'path':path,'recorded_sha256':expected,'actual_sha256':actual,'matches':actual==expected})
    snapshot={'observed_at_utc':datetime.now(timezone.utc).isoformat(),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'scope':'e_bike run inventory, result pointers and 17/26 failure audit. Read-only scan; no moves/deletions/releases. Concurrent-process inventory was unavailable; artifact states are file observations, not proof of idle processes.',
              'results':results,'runs':runs,'root_files':[file_record(f) for f in sorted((ROOT/'runs/e_bike').iterdir()) if f.is_file()],
              'result_references':ref_records,'historical_path_identifiers':[{'identifier':p,'currently_exists':(ROOT/p).exists()} for p in sorted(historical_identifiers)],
              'recorded_hash_checks':hash_checks,'reference_failure_audit':failure_audit()}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'runs':len(runs),'tensor_artifact_roles':dict(Counter(p['role'] for r in runs for p in r['tensor_artifacts'])),'bytes':sum(r['bytes'] for r in runs),'results':len(results),'missing_references':[r['path'] for r in ref_records if not r['exists']],'hash_mismatches':[r['path'] for r in hash_checks if not r['matches']],'failure_categories':snapshot['reference_failure_audit']['failure_categories']},ensure_ascii=False))


if __name__=='__main__':main()
