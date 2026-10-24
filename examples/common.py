"""Helpers shared by the examples: image loading and PSNR."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(folder):
    """Grayscale images of a folder, in [0, 1]."""
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".png", ".jpg", ".bmp", ".tif"))
    return [torch.from_numpy(np.asarray(Image.open(p).convert("L"), np.float32) / 255.0).to(DEVICE) for p in files]


def psnr(x, ref):
    return float(10 * torch.log10(1.0 / ((x.clamp(0, 1) - ref) ** 2).mean()))
