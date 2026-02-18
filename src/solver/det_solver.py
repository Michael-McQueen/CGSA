"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime
import math
import torch 
from pathlib import Path
from ..misc import dist_utils, profiler_utils, print_per_class_ap50

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate, train_one_epoch_with_self_training
from ..optim import ModelEMA
from copy import deepcopy


class DetSolver(BaseSolver):
    def get_threshold(self, e):
        if e < self.cfg.burn_epochs:
            return None
        step  = e - self.cfg.burn_epochs
        total = self.cfg.epoches - self.cfg.burn_epochs
        if self.cfg.thresholdmethod == 'fixed':
            return self.cfg.pseudo_threshold
        if self.cfg.thresholdmethod == 'exp':
            return self.cfg.t_min + (self.cfg.t_max - self.cfg.t_min) * math.exp(-self.cfg.beta * step)
        if self.cfg.thresholdmethod == 'cos':
            cos_val = 0.5 * (1 + math.cos(math.pi * step / total))
            return self.cfg.t_min + (self.cfg.t_max - self.cfg.t_min) * cos_val
        if self.cfg.thresholdmethod == 'sigmoid':
            x     = step / total                 # 0 → 1
            k     = 10   # steepness coefficient, can be added to cfg
            # ------- Sigmoid formula -------
            sigmoid = 1.0 / (1.0 + math.exp(-k * (x - 0.5)))   # ∈(0,1)
            return self.cfg.t_min + (self.cfg.t_max - self.cfg.t_min) * sigmoid
        
    def fit(self, ):
        
        use_teacher: bool = self.cfg.use_teacher
        use_HSA: bool = self.cfg.use_HSA
        if use_teacher:
            teacher_ema: ModelEMA | None = None        # teacher (EMA wrapper)
            teacher_ready: bool = False

            burn_epochs = self.cfg.burn_epochs
            pseudo_threshold = self.cfg.pseudo_threshold
            tresholdmethod = self.cfg.thresholdmethod
            ema_decay_teacher = self.cfg.ema_decay_teacher
            nms_iou_threshold = self.cfg.nms_iou_threshold
            teacher_path = self.cfg.teacher_path
            teacher_path = Path(teacher_path)

        
            print(f'Burn-in epochs: {burn_epochs}')
            print(f'Pseudo threshold: {pseudo_threshold}')
            print(f'Threshold method: {tresholdmethod}')
            print(f'Teacher EMA decay: {ema_decay_teacher}')
            print(f'NMS IoU threshold: {nms_iou_threshold}')
            print(f'Teacher path: {teacher_path}')
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1

        if use_teacher:
            teacher_ema = ModelEMA(self.model, decay=self.cfg.ema_decay_teacher)

            # Only rank-0 actually reads the file 
            if dist_utils.is_main_process():
                # best_ckpt_path = self.output_dir / 'best.pth'
                
                best_ckpt_path = teacher_path

                if best_ckpt_path.is_file():
                    ckpt = torch.load(  # torch 2.3+ recommends adding weights_only=True
                        best_ckpt_path, map_location='cpu', weights_only=False
                    )
                    state_dict = ckpt.get('ema_model', ckpt.get('model', ckpt))
                    msg = teacher_ema.module.load_state_dict(state_dict, strict=True)
                    print(f'[INFO][rank-0] Teacher loaded from {best_ckpt_path}: {msg}')
                else:
                    print(f'[WARN][rank-0] {best_ckpt_path} not found → use student weights.')
            # All ranks wait for rank-0 to finish reading, then sync weights 
            if dist_utils.is_dist_available_and_initialized():
                torch.distributed.barrier()              # wait for rank-0 to complete load
                for p in teacher_ema.module.state_dict().values():
                    torch.distributed.broadcast(p, src=0)

            teacher_ready = True
            if dist_utils.is_main_process():
                print(f'[INFO] Teacher EMA initialised')


        print(f"start_epoch: {start_epcoch}, total_epochs: {args.epoches}")

        test_stats, coco_evaluator = evaluate(
            self.model, 
            self.criterion, 
            self.dinosaur,
            self.vis_encoder,
            self.postprocessor, 
            self.val_dataloader, 
            self.evaluator, 
            self.device
        )
        per_cls_ap50 = print_per_class_ap50(
            coco_evaluator.coco_eval["bbox"],   # 1st parameter: eval results
            self.evaluator.coco_gt                # 2nd parameter: COCO-API (contains class names)
        )

        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            if use_teacher==False:
                train_stats = train_one_epoch(
                    self.model, 
                    self.criterion, 
                    self.dinosaur,
                    self.vis_encoder,
                    self.train_dataloader, 
                    self.optimizer, 
                    self.device, 
                    epoch, 
                    max_norm=args.clip_max_norm, 
                    print_freq=args.print_freq, 
                    ema=self.ema, 
                    scaler=self.scaler, 
                    lr_warmup_scheduler=self.lr_warmup_scheduler,
                    writer=self.writer
                )
            else:
                score_thr = self.get_threshold(epoch)      # calculate threshold
                print(f'epoch {epoch} score_thr: {score_thr:.4f}')
                # ==== Self-training phase with Teacher guidance ====
                train_stats = train_one_epoch_with_self_training(
                    self.model,
                    teacher_ema.module,
                    self.criterion,
                    self.dinosaur,
                    self.vis_encoder,
                    self.train_dataloader,
                    self.optimizer,
                    self.device,
                    epoch,
                    max_norm=args.clip_max_norm,
                    print_freq=args.print_freq,
                    ema=self.ema,
                    scaler=self.scaler,
                    lr_warmup_scheduler=self.lr_warmup_scheduler,
                    writer=self.writer,
                    pseudo_threshold=score_thr,
                    nms_iou=nms_iou_threshold,
                )

                # ==== Update Teacher via EMA ====
                teacher_ema.update(self.model)



            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, 
                self.criterion, 
                self.dinosaur,
                self.vis_encoder,
                self.postprocessor, 
                self.val_dataloader, 
                self.evaluator, 
                self.device
            )
            per_cls_ap50 = print_per_class_ap50(
                coco_evaluator.coco_eval["bbox"],   # 1st parameter: eval results
                self.evaluator.coco_gt                # 2nd parameter: COCO-API (contains class names)
            )

            # TODO 
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
            
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.dinosaur, self.vis_encoder, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
