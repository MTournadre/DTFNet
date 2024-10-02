#!/bin/sh

LOG_DIR=runs_trajs
MODEL_NAME=test_simple_no-aug
DATA_ROOT=/home/marc.tournadre/Datasets/kubric-dtf
DEVICE="cuda:0"
#DEVICE="cuda:0 cuda:1"

#CUBLAS_WORKSPACE_CONFIG=:4096:8
#    --reproducible \
#    --grad_accumulate 4 \
#    --clip_gradient 5. \
#    --recover_from debug_checkpoint.pt \
python train.py \
    --network_type small_dtfnet \
    --model_name ${MODEL_NAME} \
    --datasets_root $DATA_ROOT \
    --stage movi_e \
    --val_datasets movi_e \
    --image_size 256 256 \
    --downscale 1 \
    --batch_size 2 \
    --traj_length 8 \
    --traj_weight 1 \
    --corr_weight 0 \
    --vis_weight 3 \
    --lap_weight 5 \
    --no-mixed_precision \
    --epsilon 1e-8 \
    --learning_rate 2e-4 \
    --wdecay 1e-7 \
    --log_dir $LOG_DIR \
    --num_steps 500000 \
    --warmup_steps 10000 \
    --constant_steps 80000 \
    --loss_type huber \
    --gamma 0.8 \
    --seed 123 \
    --device $DEVICE \
    --num_workers 8 \
    || exit 1
