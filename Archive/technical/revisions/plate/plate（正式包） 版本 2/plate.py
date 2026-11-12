"""Province car-plate detector and two independent OCR readers; no ROS dependency."""
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


def normalize_text(text):
    return ''.join(ch for ch in str(text).upper()
                   if ch.isalnum() or '\u4e00' <= ch <= '\u9fff')


def load_charsets(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    positions = data.get('positions')
    if not isinstance(positions, list) or len(positions) != 7:
        raise ValueError('charset positions must contain exactly seven entries')
    result = []
    for position in positions:
        group = data.get(position)
        if not isinstance(group, list) or not group or any(not isinstance(c, str) or len(c) != 1 for c in group):
            raise ValueError('invalid charset group: {}'.format(position))
        if len(group) != len(set(group)):
            raise ValueError('duplicate characters in charset group: {}'.format(position))
        result.append(list(group))
    return result


def create_cnn(charset_lengths, architecture='legacy'):
    """Seven-head CNN architecture used by the provincial checkpoint."""
    import torch
    from torch import nn

    class LegacyPlateOCR(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.AdaptiveAvgPool2d((1, 7)))
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.1), nn.Linear(256, count))
                for count in charset_lengths])

        def forward(self, image):
            features = self.backbone(image).squeeze(2)
            global_features = features.mean(dim=2)
            return [head(torch.cat([features[:, :, index], global_features], dim=1))
                    for index, head in enumerate(self.heads)]

    if architecture == 'legacy':
        return LegacyPlateOCR()
    if architecture != 'spatial_v2':
        raise ValueError('unknown plate CNN architecture: {}'.format(architecture))

    class SpatialPlateOCR(nn.Module):
        """Preserve upper/lower character features with a lightweight residual block."""
        def __init__(self):
            super().__init__()
            legacy = LegacyPlateOCR()
            self.backbone = nn.Sequential(*list(legacy.backbone.children())[:-1])
            self.refine = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256))
            # Start as an identity residual branch, preserving the warm-start baseline.
            nn.init.zeros_(self.refine[-1].weight)
            self.pool = nn.AdaptiveAvgPool2d((2, 7))
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(768, 256), nn.ReLU(True), nn.Dropout(0.1), nn.Linear(256, count))
                for count in charset_lengths])

        def forward(self, image):
            features = self.backbone(image)
            features = self.pool(features + self.refine(features))
            global_features = features.mean(dim=(2, 3))
            return [head(torch.cat([features[:, :, 0, index], features[:, :, 1, index], global_features], dim=1))
                    for index, head in enumerate(self.heads)]

    return SpatialPlateOCR()


class CNNReader:
    def __init__(self, weights, charset_file, device='cpu'):
        import torch
        self.torch = torch
        self.charsets = load_charsets(charset_file)
        self.device = torch.device(device)
        checkpoint = torch.load(str(weights), map_location='cpu', weights_only=True)
        if [list(group) for group in checkpoint.get('charsets', [])] != self.charsets:
            raise ValueError('CNN checkpoint and configured character order differ')
        self.image_size = tuple(checkpoint.get('image_size', (224, 64)))
        if len(self.image_size) != 2 or any(type(v) is not int or v <= 0 for v in self.image_size):
            raise ValueError('CNN image_size must contain positive width and height')
        self.architecture = checkpoint.get('architecture', 'legacy')
        self.model = create_cnn([len(group) for group in self.charsets], self.architecture)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.to(self.device).eval()

    def read(self, crop):
        import cv2
        with self.torch.inference_mode():
            rgb = cv2.cvtColor(cv2.resize(crop, self.image_size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(self.device)
            probabilities = [self.torch.softmax(output, dim=1) for output in self.model(tensor)]
            indices = [int(prob.argmax(dim=1).item()) for prob in probabilities]
            text = ''.join(self.charsets[index][value] for index, value in enumerate(indices))
            confidence = float(self.torch.stack([prob.max(dim=1).values for prob in probabilities]).mean().item())
            return text, confidence


class PaddleReader:
    def __init__(self, model_dir, device='cpu', model_name='PP-OCRv4_mobile_rec'):
        model_dir = Path(model_dir)
        for name in ('inference.json', 'inference.pdiparams', 'inference.yml'):
            if not (model_dir / name).is_file():
                raise FileNotFoundError('local Paddle model file missing: {}'.format(model_dir / name))
        from paddleocr import TextRecognition
        self.model = TextRecognition(model_name=model_name, model_dir=str(model_dir), device=device)

    def read(self, crop):
        result = next(iter(self.model.predict(input=crop, batch_size=1)), None)
        if result is None:
            return '', 0.0
        data = result if isinstance(result, dict) else getattr(result, 'json', {})
        data = data() if callable(data) else data
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            data = dict(result)
        data = data.get('res', data)
        return str(data.get('rec_text', '')), float(data.get('rec_score', 0.0))


@dataclass(frozen=True)
class OCRResult:
    status: str
    text: str
    decision: str
    cnn_text: str
    cnn_confidence: float
    paddle_text: str
    paddle_confidence: float
    error: str = ''

    def to_dict(self):
        return asdict(self)


class DualPlateOCR:
    """Run both OCR models on the same crop, preserving their individual outputs."""
    def __init__(self, cnn, paddle, charsets, paddle_min_confidence=0.70, policy='agreement_only'):
        if cnn is None or paddle is None:
            raise ValueError('both CNN and Paddle readers are required')
        self.cnn, self.paddle, self.charsets = cnn, paddle, charsets
        self.paddle_min_confidence = float(paddle_min_confidence)
        if not 0.0 <= self.paddle_min_confidence <= 1.0:
            raise ValueError('paddle_min_confidence must be between 0 and 1')
        if policy not in ('agreement_only', 'province_baseline'):
            raise ValueError('unknown arbitration policy')
        self.policy = policy

    def valid(self, text):
        return len(text) == len(self.charsets) == 7 and all(
            char in self.charsets[index] for index, char in enumerate(text))

    def read(self, crop):
        cnn_text = paddle_text = ''
        cnn_conf = paddle_conf = 0.0
        errors = []
        try:
            cnn_text, cnn_conf = self.cnn.read(crop)
            cnn_text = normalize_text(cnn_text)
        except Exception as error:
            errors.append('cnn: {}'.format(error))
        try:
            paddle_text, paddle_conf = self.paddle.read(crop)
            paddle_text = normalize_text(paddle_text)
        except Exception as error:
            errors.append('paddle: {}'.format(error))
        if errors:
            return OCRResult('error', '', 'reader_error', cnn_text, float(cnn_conf),
                             paddle_text, float(paddle_conf), '; '.join(errors))
        if self.valid(cnn_text) and cnn_text == paddle_text:
            return OCRResult('ok', cnn_text, 'agreement', cnn_text, float(cnn_conf), paddle_text, float(paddle_conf))
        if not self.valid(cnn_text) and not self.valid(paddle_text):
            return OCRResult('unknown', '', 'invalid_outputs', cnn_text, float(cnn_conf), paddle_text, float(paddle_conf))
        if self.policy == 'agreement_only':
            return OCRResult('unknown', '', 'disagreement', cnn_text, float(cnn_conf), paddle_text, float(paddle_conf))
        # Matches the provincial override rule: compare Paddle's own threshold only.
        if self.valid(paddle_text) and float(paddle_conf) >= self.paddle_min_confidence:
            return OCRResult('ok', paddle_text, 'paddle_override', cnn_text, float(cnn_conf), paddle_text, float(paddle_conf))
        if self.valid(cnn_text):
            return OCRResult('ok', cnn_text, 'paddle_rejected', cnn_text, float(cnn_conf), paddle_text, float(paddle_conf))
        return OCRResult('unknown', '', 'invalid_outputs', cnn_text, float(cnn_conf), paddle_text, float(paddle_conf))


class PlateRecognizer:
    """Detect plates, then send each crop to both OCR readers."""
    def __init__(self, detector, ocr, device='cpu', image_size=960, confidence=0.35, iou=0.70, padding=0.08,
                 max_detections=10):
        self.detector, self.ocr, self.device = detector, ocr, device
        self.image_size, self.confidence, self.iou, self.padding = image_size, confidence, iou, padding
        self.max_detections = max_detections

    def infer(self, frame):
        import numpy as np
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError('expected a nonempty BGR uint8 HxWx3 image')
        result = self.detector.predict(frame, device=self.device, imgsz=self.image_size,
                                       conf=self.confidence, iou=self.iou, max_det=self.max_detections, verbose=False)[0]
        detections = []
        if result.boxes is None:
            return detections
        height, width = frame.shape[:2]
        for box in result.boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            px, py = int((x2-x1)*self.padding), int((y2-y1)*self.padding)
            bounds = (max(0, x1-px), max(0, y1-py), min(width, x2+px), min(height, y2+py))
            a, b, c, d = bounds
            crop = frame[b:d, a:c]
            if crop.size:
                detections.append({'bbox': bounds, 'detector_confidence': float(box.conf[0].item()),
                                   'ocr': self.ocr.read(crop).to_dict()})
        return detections


def confirm_plate_samples(samples, sample_window=1, required_matches=1):
    """Apply a configured vote to caller-supplied frame detection lists.

    Callers collect frames and retain originals. This function neither captures
    images nor performs navigation, recording, output or speech.
    """
    if type(sample_window) is not int or type(required_matches) is not int or not 1 <= required_matches <= sample_window:
        raise ValueError('required_matches must be within sample_window')
    valid = []
    for sample_index, detections in enumerate(samples):
        if len(valid) >= sample_window:
            break
        if len(detections) != 1:
            continue
        ocr = detections[0]['ocr']
        text = normalize_text(ocr.get('text', ''))
        if ocr.get('status') == 'ok' and len(text) == 7:
            score_key = 'paddle_confidence' if ocr.get('decision') == 'paddle_override' else 'cnn_confidence'
            valid.append((text, sample_index, float(ocr.get(score_key, 0.0))))
    counts = Counter(item[0] for item in valid)
    if not counts:
        return {'status': 'not_confirmed', 'text': '', 'votes': {}, 'selected_index': None}
    text, matches = counts.most_common(1)[0]
    if matches < required_matches:
        return {'status': 'not_confirmed', 'text': '', 'votes': dict(counts), 'selected_index': None}
    chosen = max((item for item in valid if item[0] == text), key=lambda item: item[2])
    return {'status': 'ok', 'text': text, 'votes': dict(counts), 'selected_index': chosen[1]}


def format_plate_text(text):
    canonical = normalize_text(text)
    return canonical[:2] + '·' + canonical[2:] if len(canonical) == 7 else canonical


def load_recognizer(package_dir, device=None):
    """Load all three local models once; missing dependencies or models fail explicitly."""
    from ultralytics import YOLO
    root = Path(package_dir)
    config = json.loads((root / 'config/plate/inference.json').read_text(encoding='utf-8'))
    device = device or config['device']
    cnn_cfg, paddle_cfg, detector_cfg = config['cnn'], config['paddle'], config['detector']
    detector_path = root / detector_cfg['weights']
    if not detector_path.is_file():
        raise FileNotFoundError('local plate detector model missing: {}'.format(detector_path))
    cnn = CNNReader(root / cnn_cfg['weights'], root / cnn_cfg['charset'], device)
    paddle_device = 'gpu:0' if str(device).startswith('cuda') else 'cpu'
    paddle = PaddleReader(root / paddle_cfg['model_dir'], paddle_device, paddle_cfg['model_name'])
    ocr = DualPlateOCR(cnn, paddle, cnn.charsets, paddle_cfg['min_confidence'], config['arbitration']['policy'])
    return PlateRecognizer(YOLO(str(detector_path)), ocr, device=device,
                           image_size=detector_cfg['image_size'], confidence=detector_cfg['confidence'],
                           iou=detector_cfg['iou'], padding=detector_cfg['crop_padding'],
                           max_detections=detector_cfg['max_detections'])
