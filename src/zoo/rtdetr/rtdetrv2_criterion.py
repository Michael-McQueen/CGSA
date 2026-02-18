"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import torch 
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F 
import torchvision
import scipy
import copy

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register
    
@register()
class RTDETRCriterionv2(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher, 
        weight_dict, 
        losses, 
        alpha=0.2, 
        gamma=2.0, 
        num_classes=80, 
        boxes_weight_format=None,
        share_matched_indices=False):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            boxes_weight_format: format for boxes weight (iou, )
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.losses.append('reconstruction')
        self.register_buffer("dataset_prototypes", torch.zeros(self.num_classes, 256))
        self.register_buffer("prototype_counts", torch.zeros(self.num_classes))
        self.temperature = 0.07
        self.slot_proj = nn.Linear(768, 256, bias=False)   # 768→256
        self.delta_t_log_path = "delta_t_log.csv"
        self.delta_t_step = 0
        self.delta_t_epoch = 0


    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def loss_reconstruction(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        assert 'recon_l2' in outputs, "Model outputs should contain 'recon_l2'"
        assert 'features' in outputs, "Model outputs should contain 'features'"
        if outputs['recon_l2'] is None or outputs['features'] is None:
            return {}

        recon_l2 = outputs['recon_l2']
        features = outputs['features'].detach()  # do not compute gradient
        loss_reconstruction = F.mse_loss(recon_l2, features)

        return {'loss_reconstruction_l2': loss_reconstruction}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'reconstruction': self.loss_reconstruction, 
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    @staticmethod
    def _assign_slots_by_similarity(q_feats, slot_feats, use_hungarian=False):
        """
        Args:
            q_feats   : (B, N, D)  # decoder query features
            slot_feats: (B, S, D)  # weighted slots
        Returns:
            chosen_slots : (B, N, D)
            slot_indices : (B, N)  # int64
        """
        import scipy
        q_norm   = F.normalize(q_feats,   dim=-1)
        s_norm   = F.normalize(slot_feats, dim=-1)
        sim      = torch.einsum('bnd,bsd->bns', q_norm, s_norm)   # (B,N,S)

        slot_idx = sim.argmax(dim=-1)          # (B,N)

        if use_hungarian:                      # 2) then use Hungarian algorithm to refine one-to-one matching for 25 items
            for b in range(sim.size(0)):
                cost = -sim[b].detach().cpu().numpy()      # (N,S)

                # linear_sum_assignment only returns 25 pairs of row-col
                row, col = scipy.optimize.linear_sum_assignment(cost)
                #   row is already sorted as a subset of 0...N-1 with length 25
                slot_idx[b, torch.as_tensor(row, device=slot_idx.device)] = \
                    torch.as_tensor(col, device=slot_idx.device)

        # gather to get slot vectors
        idx_exp = slot_idx.unsqueeze(-1).expand(-1, -1, slot_feats.size(-1))
        chosen  = slot_feats.gather(1, idx_exp)        # (B,N,D)
        return chosen, slot_idx
    
    def loss_contrastive_das(self, outputs, targets, indices):
        """
        Contrastive loss (DAS)
        """
        device = outputs["pred_logits"].device
        C      = self.num_classes
        D      = outputs["features_tgt"].size(-1)

        # 1. slot assignment
        q_feats    = outputs["features_tgt"]                  # (B,N,D)
        slot_feats = self.slot_proj(outputs["weighted_slots_l2"])  # (B,S,D)
        slot_feats = F.normalize(slot_feats, dim=-1)          # normalization helps stability
        chosen_slot, _ = self._assign_slots_by_similarity(
            q_feats, slot_feats, use_hungarian=True)          # (B,N,D)

        feat_tgt  = q_feats        # decoder path
        feat_slot = chosen_slot    # slot   path
        logits    = outputs["pred_logits"]

        #-------------------------------------------------
        # 2.  P_ST (tgt) & P_SS (slot)
        #-------------------------------------------------
        P_ST = torch.zeros(C, D, device=device)
        P_SS = torch.zeros(C, D, device=device)
        cnt_src = torch.zeros(C, device=device)

        for img_i, (I, J) in enumerate(indices):
            if I.numel() == 0:
                continue
            lbl = targets[img_i]["labels"][J]
            # filter invalid class numbers (safety)
            lbl = lbl[(0 <= lbl) & (lbl < C)]
            if lbl.numel() == 0:
                continue

            ft = feat_tgt [img_i, I[:lbl.size(0)]]  # (Mi,D) decoder
            fs = feat_slot[img_i, I[:lbl.size(0)]]  # (Mi,D) slot
            for cls in lbl.unique():
                m = (lbl == cls)
                P_ST[cls] += ft[m].mean(0)
                P_SS[cls] += fs[m].mean(0)
                cnt_src[cls] += 1

        mask_src = cnt_src > 0
        if mask_src.any():
            P_ST[mask_src] /= cnt_src[mask_src].unsqueeze(-1)
            P_SS[mask_src] /= cnt_src[mask_src].unsqueeze(-1)

        #-------------------------------------------------
        # 4. EMA update dataset-level prototype  (P_ST + P_TT)
        #-------------------------------------------------
        if (not hasattr(self, 'dataset_prototypes')
                or self.dataset_prototypes.shape[1] != D):
            self.dataset_prototypes = torch.zeros(C, D, device=device)

        momentum = 0.9
        update = (cnt_src > 0)                # at least one is valid
        proto_new = torch.where(cnt_src.unsqueeze(-1) > 0, P_ST, 0.)
        proto_new = torch.where(update.unsqueeze(-1), proto_new, 0.)
        self.dataset_prototypes[update] = (
            momentum * self.dataset_prototypes[update] +
            (1 - momentum) * proto_new[update].detach()
        )

        #-------------------------------------------------
        # 5. InfoNCE: slot-path (P_SS & P_TS) vs dataset_proto
        #-------------------------------------------------
        losses = {}
        if (cnt_src.sum()) > 0:                # any domain has samples
            p_SS = F.normalize(P_SS, dim=-1)
            p_set = F.normalize(self.dataset_prototypes, dim=-1)

            loss, valid = 0., 0
            temp = self.temperature

            for cls in range(C):
                # source domain slot
                if cnt_src[cls] > 0:
                    pos = torch.exp((p_SS[cls] @ p_set[cls]) / temp)
                    neg = torch.exp((p_SS @ p_set[cls])).sum()
                    loss += -torch.log(pos / (neg + 1e-8)); valid += 1

            if valid:
                loss = loss / valid
                losses["loss_contrastive"] = (
                    loss * self.weight_dict["loss_contrastive"]
                )
        return losses    

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Retrieve the matching between the outputs of the last layer and the targets
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']

        # print("yestyes0")

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)            
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # print("yestyes1")

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                for loss in self.losses:
                    if loss == 'reconstruction' and 'recon_l2' not in aux_outputs:
                        continue  
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)\
                    
        # print("yestyes2")

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss == 'reconstruction' and 'recon_l2' not in aux_outputs:
                        continue  
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # print("yestyes3")

        # In case of encoder auxiliary losses. For rtdetr v2
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, targets)
                indices = matched['indices']
                for loss in self.losses:
                    if loss == 'reconstruction' and 'recon_l2' not in aux_outputs:
                        continue  
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            
            if class_agnostic:
                self.num_classes = orig_num_classes

        # print("yestyes4")


        if 'features_tgt' in outputs and outputs["weighted_slots_l2"] is not None:
            losses.update(self.loss_contrastive_das(outputs, targets, indices))

        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', ):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices
