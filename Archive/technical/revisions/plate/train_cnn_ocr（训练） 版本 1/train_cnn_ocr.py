#!/usr/bin/env python3
"""Fine-tune the provincial seven-position CNN from labelled plate crops."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'ws/src/competition_perception/src'))

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from competition_perception.plate import create_cnn, load_charsets


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


class PlateDataset(Dataset):
    """Read the fine-tuning project's OCR JSON annotation contract."""
    def __init__(self, path, charsets, size, augmentation=None):
        document = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if document.get('schema_version') != '1.0' or document.get('task') != 'license_plate_recognition':
            raise ValueError('unsupported OCR annotation schema: {}'.format(path))
        self.root = (Path(path).parent / document.get('image_root', '.')).resolve()
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
    with torch.inference_mode():
        for images, target in loader:
            outputs = model(images.to(device))
            predictions = torch.stack([out.argmax(dim=1) for out in outputs], dim=1).cpu()
            matches = predictions.eq(target)
            correct_plates += int(matches.all(dim=1).sum().item())
            correct_chars += int(matches.sum().item())
            total += target.shape[0]
    return {'plate_accuracy': correct_plates / total, 'character_accuracy': correct_chars / (total * 7)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='lab/features/plate/training/config/cnn_ocr.json')
    args = parser.parse_args()
    cfg = json.loads(resolve(args.config).read_text(encoding='utf-8'))
    seed = int(cfg['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if cfg['device'] == 'auto' else cfg['device'])
    charsets = load_charsets(resolve(cfg['charset']))
    checkpoint = torch.load(str(resolve(cfg['weights'])), map_location='cpu', weights_only=True)
    if [list(group) for group in checkpoint['charsets']] != charsets:
        raise ValueError('checkpoint character order differs from configuration')
    size = tuple(checkpoint['image_size'])
    model = create_cnn([len(group) for group in charsets])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)
    train = PlateDataset(resolve(cfg['train_annotations']), charsets, size, cfg.get('augmentation'))
    val = PlateDataset(resolve(cfg['val_annotations']), charsets, size)
    train_loader = DataLoader(train, batch_size=int(cfg['batch_size']), shuffle=True, num_workers=int(cfg['workers']))
    val_loader = DataLoader(val, batch_size=int(cfg['batch_size']), shuffle=False, num_workers=int(cfg['workers']))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']), weight_decay=float(cfg['weight_decay']))
    epochs, warmup = int(cfg['epochs']), int(cfg.get('warmup_epochs', 0))

    def lr_factor(epoch_index):
        if warmup and epoch_index < warmup:
            return float(epoch_index + 1) / warmup
        progress = (epoch_index - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_factor)
    mixed_precision = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=mixed_precision)
    criterion = nn.CrossEntropyLoss()
    best, stale = -1.0, 0
    output = resolve(cfg['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=mixed_precision):
                outputs = model(images)
                loss = sum(criterion(out, targets[:, i]) for i, out in enumerate(outputs))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        metrics = evaluate(model, val_loader, device)
        scheduler.step()
        print('epoch={} plate_accuracy={:.4f} character_accuracy={:.4f}'.format(
            epoch, metrics['plate_accuracy'], metrics['character_accuracy']), flush=True)
        if metrics['plate_accuracy'] > best:
            best, stale = metrics['plate_accuracy'], 0
            torch.save({'model': model.state_dict(), 'charsets': charsets, 'image_size': list(size),
                        'epoch': epoch, 'metrics': metrics}, str(output))
        else:
            stale += 1
            if stale >= int(cfg['early_stopping_patience']):
                break


if __name__ == '__main__':
    main()
