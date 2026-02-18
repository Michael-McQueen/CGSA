#!/bin/bash

set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh   # Modify to your own path
conda activate cgsa   # Change to your actual environment

torchrun --master_port=9911 --nproc_per_node=4 tools/train.py   -t importantpt/rtdetrv2_r50vd_6x_coco_ema.pth -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco_c2bdd  .yml    --use-amp --seed=0   2>&1 | tee out/output_c2bdd.log

cp output/rtdetrv2_r50vd_6x_coco_c2bdd/best.pth output/importantpt/best_c2bdd.pth

torchrun --master_port=9911 --nproc_per_node=4 tools/train.py   -t output/importantpt/best_c2bdd.pth -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco_self_training_c2bdd.yml    --use-amp --seed=0   2>&1 | tee out/output_c2bdd_s.log
