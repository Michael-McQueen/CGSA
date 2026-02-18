"""Swin-Transformer backbone wrapper built on top of timm."""

from __future__ import annotations

import importlib
from typing import Iterable, List, Sequence

import torch.nn as nn

from ...core import register

__all__ = ["SwinBackbone"]


@register()
class SwinBackbone(nn.Module):
    """Wrap timm Swin models so they can drop in as the RT-DETR backbone."""

    def __init__(
        self,
        model_name: str = "swin_large_patch4_window12_384_in22k",
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        img_size: int = 640,
        drop_path_rate: float = 0.2,
        out_indices: Sequence[int] = (1, 2, 3),
        out_channels: Sequence[int] = (512, 1024, 2048),
    ) -> None:
        super().__init__()

        if len(out_indices) != len(out_channels):
            raise ValueError("out_indices and out_channels must be the same length")

        timm = importlib.import_module("timm")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            out_indices=tuple(out_indices),
            checkpoint_path=checkpoint_path or "",
        )
        self._enable_dynamic_patch_embed(self.backbone)

        feature_info = self.backbone.feature_info
        self.source_channels: List[int] = list(feature_info.channels())
        self.source_strides: List[int] = list(feature_info.reduction())

        self.out_channels = list(out_channels)
        self.out_strides = self.source_strides

        self.proj_layers = nn.ModuleList(
            [
                nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, kernel_size=1)
                for in_c, out_c in zip(self.source_channels, self.out_channels)
            ]
        )

    def _enable_dynamic_patch_embed(self, module):
        for child in module.modules():
            cls_name = child.__class__.__name__.lower()
            if hasattr(child, "dynamic_img_size") or "patchembed" in cls_name:
                try:
                    child.dynamic_img_size = True
                except AttributeError:
                    setattr(child, "dynamic_img_size", True)

    def forward(self, x):
        feats = self.backbone(x)
        outputs = []
        for idx, (proj, feat) in enumerate(zip(self.proj_layers, feats)):
            if feat.ndim == 4:
                expected_c = proj.weight.shape[1]
                if feat.shape[1] != expected_c and feat.shape[-1] == expected_c:
                    feat = feat.permute(0, 3, 1, 2).contiguous()
            outputs.append(proj(feat))
        return outputs
