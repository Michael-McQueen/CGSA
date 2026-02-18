"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
from functools import partial

from ..core import register


__all__ = [
    'DataLoader',
    'BaseCollateFunction', 
    'BatchImageCollateFuncion',
    'batch_image_collate_fn'
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch 
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)
    
    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch 

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


@register()
class BatchImageCollateFuncion(BaseCollateFunction):
    def __init__(
        self, 
        scales=None, 
        stop_epoch=None, 
    ) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        # self.interpolation = interpolation

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)

            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=sz, mode='nearest')
                raise NotImplementedError('')

        return images, targets


from .dataset._dataset import DetDataset
@register()
class DomainAdaptationDataset(DetDataset):
    __inject__ = ['target_dataset', 'transforms', 'strong_transforms', 'weak_transforms']

    def __init__(self, target_dataset, transforms=None, strong_transforms=None, weak_transforms=None):
        super().__init__()
        self.target_dataset = target_dataset
        self.transforms = transforms
        self.strong_transforms = strong_transforms
        self.weak_transforms = weak_transforms
        self._epoch = -1  # Initialize epoch
        print("len(target_dataset):", len(target_dataset))  # Must be greater than 0

    # def __len__(self):
    #     return min(len(self.source_dataset), len(self.target_dataset))
    def __len__(self):
        return len(self.target_dataset)


    def set_epoch(self, epoch):
        self._epoch = epoch
        if hasattr(self.target_dataset, 'set_epoch'):
            self.target_dataset.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch

    def __getitem__(self, idx):
        target_idx = idx % len(self.target_dataset)
        target_img, target_target = self.target_dataset[target_idx]
        

        if self.transforms:
            target_img_strong, _, _ = self.strong_transforms(target_img, {}, self)  # Unlabeled data processing
            target_img_weak, _, _ = self.weak_transforms(target_img, {}, self)  # Unlabeled data processing


        return {
            'target_img_strong': target_img_strong,
            'target_img_weak': target_img_weak,
            'target_target': target_target,
        }


@register()
class DomainAdaptationCollateFunction(BaseCollateFunction):
    def __init__(self, scales=None, stop_epoch=None):
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else float('inf')
        self._epoch = -1

    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch

    def __call__(self, items):

        target_images = torch.stack([item['target_img_strong'] for item in items])
        target_targets = [item['target_target'] for item in items]
        target_images_weak = torch.stack([item['target_img_weak'] for item in items])

        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            target_images = F.interpolate(target_images, size=sz)

        return target_images, target_targets, target_images_weak
