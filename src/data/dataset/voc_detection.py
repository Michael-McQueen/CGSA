"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
from bisect import bisect_right
from typing import Optional, Callable, Iterable, List, Sequence, Tuple, Union

import torch
import torchvision
from PIL import Image

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]


def _resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _ensure_suffixes(suffixes: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(suffixes, (list, tuple)):
        return [str(sfx) for sfx in suffixes]
    return [str(suffixes)]


@register()
class VOCDetection(torchvision.datasets.VOCDetection, DetDataset):
    __inject__ = ['transforms', ]

    def __init__(
        self,
        root: str,
        ann_file: str = "ImageSets/Main/trainval.txt",
        label_file: Optional[str] = None,
        transforms: Optional[Callable] = None,
        img_subdir: str = "JPEGImages",
        ann_subdir: str = "Annotations",
        img_suffix: Union[str, Sequence[str]] = ".jpg",
        ann_suffix: Union[str, Sequence[str]] = ".xml",
    ):
        list_path = _resolve_path(root, ann_file)
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Annotation list {list_path} does not exist.")

        with open(list_path, 'r') as f:
            lines = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        img_suffixes = _ensure_suffixes(img_suffix)
        ann_suffixes = _ensure_suffixes(ann_suffix)

        self.images: List[str] = []
        self.annotation_files: List[str] = []

        for line in lines:
            parts = line.split()
            if len(parts) == 1:
                image_id = parts[0]
                img_path = self._resolve_id_path(root, img_subdir, image_id, img_suffixes)
                ann_path = self._resolve_id_path(root, ann_subdir, image_id, ann_suffixes)
            elif len(parts) >= 2:
                img_path = _resolve_path(root, parts[0])
                ann_path = _resolve_path(root, parts[1])
            else:
                continue

            self.images.append(img_path)
            self.annotation_files.append(ann_path)

        assert len(self.images) == len(self.annotation_files)

        labels = self._load_labels(root, label_file)
        self.label_names = list(labels)
        self.label_id2name = {i: lab for i, lab in enumerate(self.label_names)}
        self.labels_map = {lab: i for i, lab in enumerate(self.label_names)}
        self.transforms = transforms

    def _load_labels(self, root: str, label_file: Optional[str]) -> Iterable[str]:
        if label_file is None:
            return VOC_CLASSES

        label_path = _resolve_path(root, label_file)
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file {label_path} does not exist.")

        with open(label_path, 'r') as f:
            labels = [lab.strip() for lab in f.readlines() if len(lab.strip()) > 0]
        return labels

    def _resolve_id_path(
        self,
        root: str,
        subdir: str,
        image_id: str,
        suffixes: Sequence[str],
    ) -> str:
        for suffix in suffixes:
            candidate = os.path.join(root, subdir, image_id + suffix)
            if os.path.exists(candidate):
                return candidate
        # fallback to first suffix even if file is missing to surface error later
        return os.path.join(root, subdir, image_id + suffixes[0])

    def __getitem__(self, index: int):
        image, target = self.load_item(index)
        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)
        return image, target

    def load_item(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        target = self._load_target(index, image.size)
        return image, target

    def _load_target(self, index: int, image_size: Tuple[int, int]):
        parsed = self.parse_voc_xml(ET_parse(self.annotation_files[index]).getroot())
        objects = parsed['annotation'].get('object', [])
        if isinstance(objects, dict):
            objects = [objects]

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for obj in objects:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            name = obj['name']
            labels.append(self.labels_map[name])
            areas.append(max(xmax - xmin, 0.) * max(ymax - ymin, 0.))
            iscrowd.append(int(obj.get('difficult', 0)))

        w, h = image_size

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        target = {
            "boxes": convert_to_tv_tensor(boxes_tensor, 'boxes', box_format='xyxy', spatial_size=[h, w]),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
            "image_id": torch.tensor([index]),
            "orig_size": torch.tensor([w, h]),
        }
        return target


@register()
class VOCConcatDataset(DetDataset):
    __inject__ = ['voc2007', 'voc2012', 'transforms']

    def __init__(self, voc2007, voc2012, transforms=None):
        super().__init__()
        self.datasets = [voc2007, voc2012]
        self.cumulative_sizes = self._compute_cumulative_sizes()
        self.transforms = transforms
        if hasattr(voc2007, 'label_names'):
            self.label_names = list(voc2007.label_names)
        if hasattr(voc2007, 'label_id2name'):
            self.label_id2name = dict(voc2007.label_id2name)

    def _compute_cumulative_sizes(self) -> List[int]:
        sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            sizes.append(total)
        return sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(f'Index {index} out of range for dataset of size {len(self)}')

        dataset_idx = bisect_right(self.cumulative_sizes, index)
        prev_size = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        sample_idx = index - prev_size
        img, target = self.datasets[dataset_idx][sample_idx]
        if self.transforms is not None:
            img, target, _ = self.transforms(img, target, self)
        return img, target

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for dataset in self.datasets:
            if hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(epoch)
