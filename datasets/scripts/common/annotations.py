"""Shared geometry-to-label formatting for single-class detection."""


def yolo_row(bbox, width, height):
    if width <= 0 or height <= 0 or len(bbox) != 4:
        raise ValueError('invalid image or box')
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('box must have positive area inside the image')
    return f'0 {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}'


def class_yolo_row(class_id, bbox, width, height):
    x0, y0, x1, y1 = bbox
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError('out-of-frame bbox')
    return f'{class_id} {(x0+x1)/(2*width):.6f} {(y0+y1)/(2*height):.6f} {(x1-x0)/width:.6f} {(y1-y0)/height:.6f}'

