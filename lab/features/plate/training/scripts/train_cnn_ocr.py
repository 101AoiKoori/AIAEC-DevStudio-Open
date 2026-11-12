#!/usr/bin/env python3
"""Fine-tune provincial or spatial CNNs using the migrated provincial training flow."""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'ws/src/aec_vision/src'))

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from aec_vision.plate import create_cnn, load_charsets


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


class PlateDataset(Dataset):
    """Read the fine-tuning project's OCR JSON annotation contract."""
    def __init__(self, path, charsets, size, augmentation=None):
        path = Path(path)
        if path.suffix == '.txt':
            self.root = path.parent.resolve()
            self.rows = []
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if line.strip():
                    image, text = line.split('\t')
                    self.rows.append({'image': image, 'text': text})
        else:
            document = json.loads(path.read_text(encoding='utf-8-sig'))
            if document.get('schema_version') != '1.0' or document.get('task') != 'license_plate_recognition':
                raise ValueError('unsupported OCR annotation schema: {}'.format(path))
            self.root = (path.parent / document.get('image_root', '.')).resolve()
            self.rows = document.get('annotations', [])
        self.charsets, self.size, self.augmentation = charsets, tuple(size), augmentation or {}
        if not self.rows:
            raise ValueError('no OCR annotations in {}'.format(path))
        for row in self.rows:
            text = row.get('text', '')
            if len(text) != 7 or any(ch not in charsets[i] for i, ch in enumerate(text)):
                raise ValueError('unsupported seven-character plate: {!r}'.format(text))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / row['image']
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('cannot decode image: {}'.format(path))
        cfg = self.augmentation
        rotation = float(cfg.get('rotation_degrees', 0))
        if rotation:
            h, w = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-rotation, rotation), 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        value = image.astype(np.float32)
        contrast, brightness = float(cfg.get('contrast', 0)), float(cfg.get('brightness', 0))
        if contrast:
            mean = value.mean()
            value = (value - mean) * random.uniform(1 - contrast, 1 + contrast) + mean
        if brightness:
            value *= random.uniform(1 - brightness, 1 + brightness)
        image = np.clip(value, 0, 255).astype(np.uint8)
        if random.random() < float(cfg.get('blur_probability', 0)):
            image = cv2.GaussianBlur(image, (3, 3), 0)
        if random.random() < float(cfg.get('jpeg_probability', 0)):
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, random.randint(55, 92)])
            if ok:
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(cv2.resize(image, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
        targets = torch.tensor([self.charsets[i].index(ch) for i, ch in enumerate(row['text'])], dtype=torch.long)
        return tensor, targets


def evaluate(model, loader, device):
    model.eval()
    correct_plates = correct_chars = total = 0
    position_correct = torch.zeros(7, dtype=torch.long)
    with torch.inference_mode():
        for images, target in loader:
            outputs = model(images.to(device))
            predictions = torch.stack([out.argmax(dim=1) for out in outputs], dim=1).cpu()
            matches = predictions.eq(target)
            correct_plates += int(matches.all(dim=1).sum().item())
            correct_chars += int(matches.sum().item())
            position_correct += matches.sum(dim=0)
            total += target.shape[0]
    return {'count': total, 'plate_accuracy': correct_plates / total,
            'character_accuracy': correct_chars / (total * 7),
            'position_accuracy': (position_correct / total).tolist()}


def initialize_model(checkpoint, counts, architecture):
    model = create_cnn(counts, architecture)
    source_architecture = checkpoint.get('architecture', 'legacy')
    if source_architecture == architecture:
        model.load_state_dict(checkpoint['model'], strict=True)
    elif source_architecture == 'legacy' and architecture == 'spatial_v2':
        source = checkpoint['model']
        target = model.state_dict()
        for key, value in source.items():
            if key.startswith('heads.') and key.endswith('.0.weight'):
                # Average the two height bins initially, reproducing the legacy local feature.
                target[key] = torch.cat([value[:, :256] / 2, value[:, :256] / 2, value[:, 256:]], dim=1)
            else:
                if key not in target or target[key].shape != value.shape:
                    raise ValueError('cannot transfer provincial parameter: {}'.format(key))
                target[key] = value
        model.load_state_dict(target, strict=True)
    else:
        raise ValueError('unsupported architecture transfer')
    return model


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='lab/features/plate/training/config/cnn_ocr.json')
    parser.add_argument('--architecture', choices=['legacy', 'spatial_v2'])
    parser.add_argument('--output', help='New best-checkpoint path; existing output directories are rejected.')
    parser.add_argument('--weights', help='Override the input checkpoint for evaluation or fine-tuning.')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    args = parser.parse_args()
    cfg = json.loads(resolve(args.config).read_text(encoding='utf-8'))
    if args.output:
        cfg['output'] = args.output
    if args.weights:
        cfg['weights'] = args.weights
    seed = int(cfg['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg.get('cpu_threads', 4)))
    cv2.setNumThreads(1)
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if cfg['device'] == 'auto' else cfg['device'])
    charsets = load_charsets(resolve(cfg['charset']))
    checkpoint = torch.load(str(resolve(cfg['weights'])), map_location='cpu', weights_only=True)
    if [list(group) for group in checkpoint['charsets']] != charsets:
        raise ValueError('checkpoint character order differs from configuration')
    size = tuple(checkpoint['image_size'])
    architecture = args.architecture or (checkpoint.get('architecture', 'legacy') if args.evaluate_only else cfg.get('architecture', 'legacy'))
    model = initialize_model(checkpoint, [len(group) for group in charsets], architecture)
    model.to(device)
    loader_options = dict(batch_size=int(cfg['batch_size']), num_workers=int(cfg['workers']),
                          pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                          persistent_workers=int(cfg['workers']) > 0)
    split = args.split if args.evaluate_only else 'val'
    sources = cfg.get(split + '_sources') or {'default': cfg[split + '_annotations']}
    validation = {name: PlateDataset(resolve(path), charsets, size) for name, path in sources.items()}
    val_loaders = {name: DataLoader(data, shuffle=False, **loader_options) for name, data in validation.items()}
    if args.evaluate_only:
        metrics = {name: evaluate(model, loader, device) for name, loader in val_loaders.items()}
        print(json.dumps({'architecture': architecture, 'weights': cfg['weights'], 'split': split, 'metrics': metrics}, ensure_ascii=False, indent=2), flush=True)
        return
    train_sources = cfg.get('train_sources') or {'default': cfg['train_annotations']}
    training = {name: PlateDataset(resolve(path), charsets, size, cfg.get('augmentation')) for name, path in train_sources.items()}
    train_paths = {(data.root / row['image']).resolve() for data in training.values() for row in data.rows}
    val_paths = {(data.root / row['image']).resolve() for data in validation.values() for row in data.rows}
    if train_paths & val_paths:
        raise ValueError('train and validation image paths overlap')
    train = ConcatDataset(list(training.values()))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, shuffle=True, generator=generator, **loader_options)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']), weight_decay=float(cfg['weight_decay']))
    epochs, warmup = int(cfg['epochs']), int(cfg.get('warmup_epochs', 0))

    def lr_factor(epoch_index):
        if warmup and epoch_index < warmup:
            return float(epoch_index + 1) / warmup
        progress = (epoch_index - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_factor)
    mixed_precision = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    criterion = nn.CrossEntropyLoss()
    output = resolve(cfg['output'])
    output.parent.mkdir(parents=True, exist_ok=False)
    baseline = {name: evaluate(model, loader, device) for name, loader in val_loaders.items()}
    best = sum(m['plate_accuracy'] for m in baseline.values()) / len(baseline)
    stale = 0
    metadata = {'config': cfg, 'architecture': architecture, 'source_weights': cfg['weights'],
                'torch_version': str(torch.__version__), 'device': str(device),
                'device_name': torch.cuda.get_device_name(device) if mixed_precision else 'CPU',
                'train_counts': {k: len(v) for k, v in training.items()},
                'val_counts': {k: len(v) for k, v in validation.items()},
                'parameter_count': sum(p.numel() for p in model.parameters()), 'initial_validation': baseline,
                'selection': 'unweighted mean of per-source validation whole-plate accuracy'}
    (output.parent / 'run_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    def save_best(epoch, metrics):
        torch.save({'model': model.state_dict(), 'charsets': charsets, 'image_size': list(size),
                    'architecture': architecture, 'epoch': epoch, 'metrics': metrics}, str(output))

    save_best(0, baseline)
    print(json.dumps({'initial_validation': baseline, 'architecture': architecture}), flush=True)
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        loss_sum = count = 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=mixed_precision):
                outputs = model(images)
                loss = sum(criterion(out, targets[:, i]) for i, out in enumerate(outputs))
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get('gradient_clip', 5.0)))
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.item()) * images.shape[0]
            count += images.shape[0]
        metrics = {name: evaluate(model, loader, device) for name, loader in val_loaders.items()}
        scheduler.step()
        score = sum(m['plate_accuracy'] for m in metrics.values()) / len(metrics)
        record = {'epoch': epoch, 'train_loss': loss_sum / count, 'metrics': metrics,
                  'selection_score': score, 'seconds': time.monotonic() - epoch_started}
        print(json.dumps(record), flush=True)
        with (output.parent / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        if score > best:
            best, stale = score, 0
            save_best(epoch, metrics)
        else:
            stale += 1
            if stale >= int(cfg['early_stopping_patience']):
                break
        if time.monotonic() - started >= float(cfg.get('max_minutes', 30)) * 60:
            print('Configured time budget reached at epoch boundary.', flush=True)
            break
    print('Finished: best validation score={:.6f}; {}'.format(best, output), flush=True)


if __name__ == '__main__':
    main()
