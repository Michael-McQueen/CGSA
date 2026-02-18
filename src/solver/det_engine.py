"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp 
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
import torchvision.transforms.functional as F
import torch.nn.functional as F1
import time

import torch
from typing import List, Tuple
from torchvision.ops import nms, box_convert

def apply_nms_to_pseudo_labels(logits, boxes, score_thresh=0.3, iou_thresh=0.5):
    """
    Apply NMS to teacher output and return processed boxes and labels.
    Args:
        logits: Tensor [num_queries, num_classes]
        boxes: Tensor [num_queries, 4], format: cxcywh
        score_thresh: threshold for filtering low-confidence boxes
        iou_thresh: IoU threshold for NMS
    Returns:
        filtered_boxes, filtered_labels
    """
    scores, labels = logits.sigmoid().max(dim=-1)  # shape: [num_queries]
    
    # Initial score filtering
    keep = scores > score_thresh
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]

    # Filter out boxes with invalid width or height
    wh = boxes[:, 2:4]
    size_ok = (wh[:, 0] > 1e-3) & (wh[:, 1] > 1e-3)
    boxes = boxes[size_ok]
    scores = scores[size_ok]
    labels = labels[size_ok]

    # Convert from cxcywh → xyxy for NMS
    boxes_xyxy = box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')

    # Perform NMS per class
    keep_indices = []
    for cls in labels.unique():
        cls = cls.item()
        cls_mask = labels == cls
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = nms(cls_boxes, cls_scores, iou_thresh)

        # Record original indices
        kept_idx = torch.nonzero(cls_mask, as_tuple=True)[0][cls_indices]
        keep_indices.append(kept_idx)
    if len(keep_indices) == 0:
        return torch.empty((0, 4), device=boxes.device), torch.empty((0,), dtype=torch.long, device=boxes.device), torch.empty((0,), device=boxes.device)

    keep_indices = torch.cat(keep_indices, dim=0)
    return boxes[keep_indices], labels[keep_indices], scores[keep_indices]

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    dinosaur, vis_encoder,  # Added Slot Attention related components
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)
    total_iter_time = 0.0
    num_iters = 0
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if device.type == "cuda":
            torch.cuda.synchronize()
        iter_start = time.perf_counter()

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)


        # **Compute Slots**
        if vis_encoder is not None and dinosaur is not None:
            feature = vis_encoder(F.resize(samples, (320, 320), antialias=True))
            recon_l2, slots, _, _, _, _, _, weighted_slots_l2 = dinosaur(feature)

        else:
            feature = None
            recon_l2 = None
            slots = None
            weighted_slots_l2 = None

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, slots, None, targets=targets)
                outputs['recon_l2'] = recon_l2  # pass to criterion
                outputs['features'] = feature  # pass to criterion
                outputs['weighted_slots_l2'] = weighted_slots_l2  # pass to criterion
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, slots, None, targets=targets)
            outputs['recon_l2'] = recon_l2  # pass to criterion
            outputs['features'] = feature  # pass to criterion
            outputs['weighted_slots_l2'] = weighted_slots_l2  # pass to criterion
            loss_dict = criterion(outputs, targets, **metas)
            
            loss : torch.Tensor = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # ====== One iteration finished, stop timing ======
        if device.type == "cuda":
            torch.cuda.synchronize()
        iter_end = time.perf_counter()

        iter_time = (iter_end - iter_start) * 1000.0  # ms
        total_iter_time += iter_time
        num_iters += 1
        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
    # ---------------- epoch finished ----------------
    avg_iter_time = total_iter_time / max(1, num_iters)  # ms

    if dist_utils.is_main_process():
        print(f"Average per-iteration time: {avg_iter_time:.2f} ms")                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_with_self_training(
    student_model, teacher_model, criterion, dinosaur, vis_encoder,
    data_loader, optimizer, device, epoch, max_norm=0, pseudo_threshold=0.3, nms_iou=0.5, **kwargs
):
    student_model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    for i, (samples, targets, target_images_weak) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)
        target_images_weak = target_images_weak.to(device)
        # 1. Teacher model generates pseudo labels 
        teacher_model.eval()
        with torch.no_grad():
            slots = None
            if vis_encoder is not None and dinosaur is not None:
                feat_tgt = vis_encoder(F.resize(target_images_weak, (320, 320), antialias=True))
                _, slots, _, _, _, _, _, _ = dinosaur(feat_tgt)
            pred_dt = teacher_model(F.resize(target_images_weak, (640, 640), antialias=True), slots, None)

        teacher_outputs = pred_dt


        pseudo_targets = []
        for logits, boxes in zip(teacher_outputs['pred_logits'], teacher_outputs['pred_boxes']):
            nms_boxes, nms_labels, nms_scores = apply_nms_to_pseudo_labels(
                logits, boxes,
                score_thresh=pseudo_threshold,
                iou_thresh=nms_iou 
            )

            # Scale to student size (optional)
            # nms_boxes = nms_boxes * torch.tensor([student_W, student_H, student_W, student_H], device=nms_boxes.device)

            pseudo_targets.append({
                "boxes": nms_boxes.detach(),
                "labels": nms_labels.detach(),
                "scores": nms_scores.detach(),
                # "orig_size": torch.tensor([student_H, student_W], device=boxes.device)
            })

        targets = pseudo_targets  # Replace original targets with pseudo labels
        # **Compute Slots**
        if vis_encoder is not None and dinosaur is not None:
            feature = vis_encoder(F.resize(samples, (320, 320), antialias=True))
            # print("feature.shape, ", feature.shape)
            recon_l2, slots, _, _, _, _, _, weighted_slots_l2 = dinosaur(feature)
            # _, slots, _, _, recon_l2, _, _, weighted_slots_l2 = dinosaur(feature)
        else:
            feature = None
            recon_l2 = None
            slots = None
            weighted_slots_l2 = None

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = student_model(samples, slots, None, targets=targets)
                outputs['recon_l2'] = recon_l2  # pass to criterion
                outputs['features'] = feature  # pass to criterion
                outputs['weighted_slots_l2'] = weighted_slots_l2  # pass to criterion
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = student_model(samples, slots, None, targets=targets)
            outputs['recon_l2'] = recon_l2  # pass to criterion
            outputs['features'] = feature  # pass to criterion
            loss_dict = criterion(outputs, targets, **metas)
            
            loss : torch.Tensor = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(student_model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        # loss_dict['reconstruction_loss_l2'] = reconstruction_scaled_l2
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, dinosaur, vis_encoder, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if vis_encoder is not None and dinosaur is not None:
            feature = vis_encoder(F.resize(samples, (320, 320), antialias=True))
            _, slots, _, _, _, _, _, _ = dinosaur(feature)
        else:
            slots = None

        outputs = model(samples, slots, None)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator



