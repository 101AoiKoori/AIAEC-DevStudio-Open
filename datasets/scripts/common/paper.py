"""Reusable paper drawing and perspective placement primitives."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .perspective import warp_patch


def name_paper(text, font_path):
    """Draw the configured text on a paper image using the supplied font."""
    image = Image.new("RGB", (600, 160), (229, 228, 223))
    font = ImageFont.truetype(str(font_path), 98)
    draw = ImageDraw.Draw(image)
    bounds = draw.textbbox((0, 0), text, font=font)
    draw.text(((600 - (bounds[2] - bounds[0])) / 2,
               (160 - (bounds[3] - bounds[1])) / 2 - bounds[1]),
              text, font=font, fill=(20, 20, 20))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def paste(image, patch, quad):
    """Paste a paper patch into the image and return its visible bounding box."""
    warped, mask, box, _ = warp_patch(patch, quad, (image.shape[1], image.shape[0]))
    image[mask > 0] = warped[mask > 0]
    return box
