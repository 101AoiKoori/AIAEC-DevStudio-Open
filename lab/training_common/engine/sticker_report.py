"""Fixed-threshold diagnostics for the 24-class waste-sticker detector."""
import json
from pathlib import Path


def sticker_class_report(rows, names):
    """Report class-specific matched instances and complete multi-sticker images."""
    if not isinstance(names, list) or len(names) != 24 or len(set(names)) != 24 or not rows:
        raise ValueError('Expected 24 distinct class names and nonempty predictions')
    classes = [{'class_id': i, 'name': name, 'ground_truth': 0, 'matched': 0,
                'false_predictions': 0, 'recall': None} for i, name in enumerate(names)]
    multi = complete = complete_no_false = 0
    negative = negative_false_images = negative_false_predictions = 0
    by_count = {str(count): {'images': 0, 'all_stickers_matched_images': 0,
                            'exactly_correct_images': 0} for count in range(5)}
    for row in rows:
        truths, predictions = row['ground_truth'], row['predictions']
        for item in truths + predictions:
            cls = item['class_id']
            if type(cls) is not int or cls not in range(24): raise ValueError('Invalid sticker class')
        for item in truths: classes[item['class_id']]['ground_truth'] += 1
        seen_gt, seen_prediction = set(), set()
        for gi, pi, _ in row['matches']:
            if gi in seen_gt or pi in seen_prediction or not 0 <= gi < len(truths) or not 0 <= pi < len(predictions):
                raise ValueError('Invalid or duplicate matching indices')
            if truths[gi]['class_id'] != predictions[pi]['class_id']: raise ValueError('Matched class mismatch')
            seen_gt.add(gi); seen_prediction.add(pi); classes[truths[gi]['class_id']]['matched'] += 1
        false = row['false_prediction_indices']
        if len(false) != len(set(false)) or set(false) != set(range(len(predictions))) - seen_prediction:
            raise ValueError('False prediction indices disagree with matches')
        for index in false: classes[predictions[index]['class_id']]['false_predictions'] += 1
        count = len(truths)
        group = by_count.setdefault(str(count), {'images': 0, 'all_stickers_matched_images': 0,
                                                'exactly_correct_images': 0})
        group['images'] += 1
        all_matched = len(seen_gt) == count
        group['all_stickers_matched_images'] += all_matched
        group['exactly_correct_images'] += all_matched and not bool(false)
        if not truths:
            negative += 1
            negative_false_images += bool(predictions)
            negative_false_predictions += len(predictions)
        if len(truths) >= 2:
            multi += 1
            if len(seen_gt) == len(truths):
                complete += 1
                complete_no_false += not bool(false)
    for item in classes:
        if item['ground_truth']: item['recall'] = item['matched'] / item['ground_truth']
        predicted = item['matched'] + item['false_predictions']
        item['precision'] = item['matched'] / predicted if predicted else None
    for group in by_count.values():
        group['exactly_correct_rate'] = (group['exactly_correct_images'] / group['images']
                                         if group['images'] else None)
    observed = [item for item in classes if item['ground_truth']]
    return {'images': len(rows), 'classes': classes,
            'observed_class_macro_recall': sum(item['recall'] for item in observed) / len(observed) if observed else None,
            'absent_class_ids': [item['class_id'] for item in classes if not item['ground_truth']],
            'multi_sticker_images': multi, 'all_stickers_matched_images': complete,
            'all_stickers_matched_without_extra_predictions_images': complete_no_false,
            'by_sticker_count': by_count, 'negative_images': negative,
            'negative_false_positive_images': negative_false_images,
            'negative_false_predictions': negative_false_predictions,
            'negative_false_positive_image_rate': negative_false_images / negative if negative else None,
            'scope': 'Fixed-threshold detector matches. Macro recall uses classes with ground truth. '
                     'Bin association and disposal belong to downstream task evaluation.'}


def write_sticker_report(evaluation_dir, names):
    evaluation_dir = Path(evaluation_dir)
    predictions = evaluation_dir / 'predictions/predictions.jsonl'
    rows = [json.loads(line) for line in predictions.read_text(encoding='utf-8').splitlines() if line.strip()]
    report = sticker_class_report(rows, names)
    summary = json.loads((evaluation_dir / 'summary.json').read_text(encoding='utf-8'))
    report['split'] = summary['split']
    path = evaluation_dir / 'waste_sticker_class_report.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != report:
            raise ValueError('Existing sticker report differs from the evaluated predictions')
    else:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path
