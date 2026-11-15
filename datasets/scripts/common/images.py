"""Shared transparent-image operations used by dataset tasks."""
import cv2
import numpy as np


def read_image(path, mode=cv2.IMREAD_COLOR):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), mode)
    if image is None:
        raise ValueError(f'cannot decode image: {path}')
    return image


def load_rgba(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f'图片必须是带透明通道的 PNG：{path}')
    pixels = np.argwhere(image[:, :, 3] > 127)
    if not len(pixels):
        raise ValueError(f'透明素材没有前景：{path}')
    y1, x1 = pixels.min(axis=0)
    y2, x2 = pixels.max(axis=0) + 1
    return image[y1:y2, x1:x2]


def resized(image, width=None, height=None):
    h, w = image.shape[:2]
    scale = width / w if width is not None else height / h
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def box(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('可见掩码为空')
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def write_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f'cannot encode {path}')
    encoded.tofile(str(path))


def exposure(image, gain, gamma):
    if not 0.78 <= gain <= 1.20 or not 0.92 <= gamma <= 1.10:
        raise ValueError('exposure outside bounded range')
    x = image.astype(np.float32) / 255
    return np.clip(np.power(np.clip(x * gain, 0, 1), gamma) * 255, 0, 255).astype(np.uint8)

