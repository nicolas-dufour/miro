"""Image transforms used by the dataset preprocessing pipeline.

Vendored from the legacy ``cad.utils.image_processing`` so the miro package
has no runtime dependency on ``cad/``.
"""
from __future__ import annotations

import torch
import torchvision


def remap_image_torch(image: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """Inverse-normalize a normalized image tensor back to uint8 [0, 255]."""
    image_torch = ((image * std) + mean) * 255.0
    return torch.clip(image_torch, 0, 255).to(torch.uint8)


class CenterCrop(torch.nn.Module):
    """Center-crop to either an explicit ``size`` or a target aspect ``ratio``.

    With ``ratio="1:1"`` (the default), this returns the largest centered
    square that fits inside the image. Accepts PIL images and tensors alike.
    """

    def __init__(self, size: int | tuple[int, int] | None = None, ratio: str = "1:1"):
        super().__init__()
        self.size = size
        self.ratio = ratio

    def forward(self, img):
        if self.size is None:
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                w, h = img.size
            num, den = self.ratio.split(":")
            ratio = float(num) / float(den)
            ratioed_w = int(h * ratio)
            ratioed_h = int(w / ratio)
            if w >= h:
                size = (ratioed_h, w) if ratioed_h <= h else (h, ratioed_w)
            else:
                size = (h, ratioed_w) if ratioed_w <= w else (ratioed_h, w)
        else:
            size = self.size
        return torchvision.transforms.functional.center_crop(img, size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, ratio={self.ratio!r})"
