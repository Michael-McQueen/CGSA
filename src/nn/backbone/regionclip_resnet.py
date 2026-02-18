"""
RegionCLIP flavored ResNet backbone (Approach A: Keep RegionCLIP CNN structure for detection).

Key implementation points:
- Uses RegionCLIP/CLIP ResNet's three-conv stem + anti-aliasing downsampling + special Bottleneck.
- Optional AttentionPool2d head (controlled by pool_vec/create_attnpool), but recommended to disable for detection.
- Only loads CNN parts from RegionCLIP checkpoint (conv1/2/3, bn1/2/3, layer1~4),
  does not load attnpool / projection head.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import FrozenBatchNorm2d, freeze_batch_norm2d
from ...core import register

__all__ = ["RegionCLIPResNet"]


class Bottleneck(nn.Module):
    """RegionCLIP variant of the standard ResNet bottleneck."""

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, norm_type: str = "frozenbn"):
        super().__init__()

        norm_type = norm_type.lower()
        norm_layer = (
            FrozenBatchNorm2d
            if norm_type == "frozenbn"
            else nn.SyncBatchNorm
            if norm_type == "syncbn"
            else None
        )
        if norm_layer is None:
            raise ValueError(f"Unsupported norm_type {norm_type}")

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        # RegionCLIP / CLIP anti-alias downsampling: AvgPool first then conv3
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride > 1 or inplanes != planes * self.expansion:
            modules = [
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, bias=False)),
                ("1", norm_layer(planes * self.expansion)),
            ]
            self.downsample = nn.Sequential(OrderedDict(modules))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    """Attention pooling head copied from CLIP."""

    def __init__(self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int | None = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spatial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> HWNC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        # Only take CLS token (the 0th one)
        return x[0]


@register()
class RegionCLIPResNet(nn.Module):
    """RegionCLIP / CLIP style ResNet backbone compatible with RegionCLIP checkpoints.

    Recommended for detection:
        pool_vec = False
        create_attnpool = False
    This way forward only returns multi-layer feature maps without global vectors.
    """

    def __init__(
        self,
        depth: int = 50,
        return_idx: Sequence[int] = (1, 2, 3),
        freeze_at: int = 0,
        input_resolution: int = 224,
        pool_vec: bool = False,
        create_attnpool: bool = False,
        norm_type: str = "frozenbn",
        pretrained: bool | str = False,
        pretrained_path: str | None = None,
        strict_load: bool = False,
        allowed_prefixes: Sequence[str] | None = None,
    ) -> None:
        super().__init__()

        assert depth in (50, 101, 200), "RegionCLIP ResNet only supports depth 50/101/200"
        stage_blocks = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            200: [4, 6, 10, 6],
        }[depth]
        width = 64 if depth in (50, 101) else 80

        self.norm_type = norm_type.lower()
        self.input_resolution = input_resolution
        self.return_idx = list(return_idx)
        self.pool_vec = pool_vec
        self.allowed_prefixes = list(allowed_prefixes) if allowed_prefixes else []

        # ------------------------
        #  Three-layer conv stem (CLIP style)
        # ------------------------
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = self._build_norm(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = self._build_norm(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = self._build_norm(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # ------------------------
        #  Residual stages (RegionCLIP Bottleneck)
        # ------------------------
        self._inplanes = width
        self.layer1 = self._make_layer(width, stage_blocks[0])
        self.layer2 = self._make_layer(width * 2, stage_blocks[1], stride=2)
        self.layer3 = self._make_layer(width * 4, stage_blocks[2], stride=2)
        self.layer4 = self._make_layer(width * 8, stage_blocks[3], stride=2)

        embed_dim = width * 32
        if self.pool_vec or create_attnpool:
            stride32_dim = max(1, input_resolution // 32)
            self.attnpool = AttentionPool2d(stride32_dim, embed_dim, width * 32 // 64, embed_dim)
        else:
            self.attnpool = None

        # Channels and strides for each stage (layer1~4)
        stage_channels = [width * 4, width * 8, width * 16, width * 32]
        stage_strides = [4, 8, 16, 32]
        self.out_channels = [stage_channels[i] for i in self.return_idx]
        self.out_strides = [stage_strides[i] for i in self.return_idx]

        self.freeze(freeze_at)

        # ------------------------
        #  Load RegionCLIP pretrained weights
        # ------------------------
        weight_path = None
        if isinstance(pretrained, str) and pretrained:
            weight_path = pretrained
        elif isinstance(pretrained, bool) and pretrained:
            weight_path = pretrained_path
        elif pretrained_path:
            weight_path = pretrained_path

        if weight_path:
            self._load_pretrained(weight_path, strict=strict_load)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _build_norm(self, num_channels: int) -> nn.Module:
        if self.norm_type == "frozenbn":
            return FrozenBatchNorm2d(num_channels)
        if self.norm_type == "syncbn":
            return nn.SyncBatchNorm(num_channels)
        raise ValueError(f"Unsupported norm_type {self.norm_type}")

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [Bottleneck(self._inplanes, planes, stride=stride, norm_type=self.norm_type)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, norm_type=self.norm_type))
        return nn.Sequential(*layers)

    def freeze(self, freeze_at: int = 0) -> None:
        def _freeze_module(module: nn.Module) -> None:
            for p in module.parameters():
                p.requires_grad = False
            freeze_batch_norm2d(module)

        if freeze_at >= 1:
            for layer in (self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3):
                _freeze_module(layer)
        stage_modules = [self.layer1, self.layer2, self.layer3, self.layer4]
        for idx, stage in enumerate(stage_modules, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    _freeze_module(block)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> List[torch.Tensor] | torch.Tensor:
        # stem
        def stem(t: torch.Tensor) -> torch.Tensor:
            for conv, bn in ((self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)):
                t = self.relu(bn(conv(t)))
            return self.avgpool(t)

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)

        outputs: List[torch.Tensor] = []
        for idx, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            if idx in self.return_idx:
                outputs.append(x)

        # Detection scenario: pool_vec=False, directly return feature maps
        if self.pool_vec and self.attnpool is not None:
            return self.attnpool(x)
        return outputs

    # ------------------------------------------------------------------ #
    # pretrained loading helpers
    # ------------------------------------------------------------------ #
    def _load_pretrained(self, weight_path: str, strict: bool = False) -> None:
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"pretrained weights {weight_path} not found")

        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for key in (
                "model",
                "state_dict",
                "ema_model",
                "ema_state_dict",
                "student",
                "teacher",
            ):
                if isinstance(ckpt.get(key), dict):
                    ckpt = ckpt[key]
                    break

        if not isinstance(ckpt, dict):
            raise ValueError(f"Unsupported checkpoint structure at {weight_path}")

        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}

        visual_keys = self._extract_visual_state(ckpt)
        missing = set(self.state_dict().keys()) - set(visual_keys.keys())
        unexpected = set(visual_keys.keys()) - set(self.state_dict().keys())
        msg = self.load_state_dict(visual_keys, strict=strict)
        print(f"[RegionCLIPResNet] Loaded weights from {weight_path}: {msg}")
        if missing and strict:
            print(f"[RegionCLIPResNet] Missing keys: {sorted(missing)[:10]} ...")
        if unexpected and strict:
            print(f"[RegionCLIPResNet] Unexpected keys: {sorted(unexpected)[:10]} ...")

    def _extract_visual_state(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Support multiple checkpoint prefixes
        prefixes: Iterable[str] = [
            "visual.",
            "clip_backbone.visual.",
            "backbone.visual.",
            "backbone.bottom_up.",
            "student.visual.",
            "teacher.visual.",
            "clip_backbone.",
            "",
        ]
        prefixes = list(self.allowed_prefixes) + list(prefixes)

        # Only want CNN backbone parts, don't take attnpool / heads
        allowed_roots = {
            "conv1",
            "bn1",
            "conv2",
            "bn2",
            "conv3",
            "bn3",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        }

        visual_state: Dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            trimmed = key
            for prefix in prefixes:
                if trimmed.startswith(prefix):
                    trimmed = trimmed[len(prefix):]
                    break
            while trimmed.startswith("visual."):
                trimmed = trimmed[len("visual."):]

            if not trimmed:
                continue

            parts = trimmed.split(".")
            root = parts[0]
            if root not in allowed_roots:
                # Find the first valid root appearing in key (e.g. state_dict.visual.layer1...)
                for idx, part in enumerate(parts):
                    if part in allowed_roots:
                        trimmed = ".".join(parts[idx:])
                        parts = trimmed.split(".")
                        root = parts[0]
                        break

            if root not in allowed_roots:
                continue
            visual_state[trimmed] = tensor
        return visual_state
