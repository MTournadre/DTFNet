#/bin/sh

# This script contains the parameters to train the presented DTF-Net.
# You will need to adapt parameters / paths according to your hardware
# and environment

# Note: Steps are given in samples, not batches.

LOG_DIR=runs
MODEL_NAME=dtfnet
DATA_ROOT=/data/kubric_dtf
DEVICES="cuda:0"
#DEVICES="$( echo ,${CUDA_VISIBLE_DEVICES} | sed 's/,/ cuda:/g' )"

# First training stage: 8 frames at 256x256 (bs=8)
python train.py \
    --network_type dtfnet \
    --model_name ${MODEL_NAME}_stage1 \
    --datasets_root $DATA_ROOT \
    --stage movi_e \
    --val_datasets movi_e \
    --image_size 256 256 \
    --batch_size 8 \
    --traj_length 8 \
    --traj_weight 1 \
    --corr_weight 0 \
    --lap_weight 0 \
    --vis_weight 3 \
    --no-mixed_precision \
    --epsilon 1e-8 \
    --learning_rate 2e-4 \
    --wdecay 1e-7 \
    --log_dir $LOG_DIR \
    --num_steps 500000 \
    --warmup_steps 10000 \
    --constant_steps 120000 \
    --loss_type huber \
    --gamma 0.8 \
    --seed 123 \
    --device $DEVICES \
    --num_workers 4 \
    || exit 1

# Second training stage: 24 frames at 192x192 (bs=4)
python train.py \
    --network_type dtf_net \
    --model_name ${MODEL_NAME}_stage2 \
    --network models/${MODEL_NAME}_stage1.pt \
    --datasets_root $DATA_ROOT \
    --stage movi_e \
    --val_datasets movi_e \
    --image_size 192 192 \
    --batch_size 4 \
    --traj_length 24 \
    --traj_weight 1 \
    --corr_weight 0 \
    --lap_weight 0 \
    --vis_weight 3 \
    --no-mixed_precision \
    --epsilon 1e-8 \
    --learning_rate 1.5e-4 \
    --wdecay 1e-7 \
    --log_dir $LOG_DIR \
    --num_steps 300000 \
    --warmup_steps 5000 \
    --constant_steps 100000 \
    --loss_type huber \
    --gamma 0.8 \
    --seed 234 \
    --device $DEVICES \
    --num_workers 4 \
    || exit 1

# Third stage: 12 frames at 256x256 (bs=4)
python train.py \
    --network_type dtf_net \
    --model_name ${MODEL_NAME}_stage3 \
    --network models/${MODEL_NAME}_stage2.pt \
    --datasets_root $DATA_ROOT \
    --stage movi_e \
    --val_datasets movi_e \
    --image_size 256 256 \
    --batch_size 8 \
    --traj_length 12 \
    --traj_weight 1 \
    --corr_weight 0 \
    --lap_weight 0 \
    --vis_weight 3 \
    --no-mixed_precision \
    --epsilon 1e-8 \
    --learning_rate 1e-4 \
    --wdecay 1e-7 \
    --log_dir $LOG_DIR \
    --num_steps 200000 \
    --warmup_steps 5000 \
    --constant_steps 80000 \
    --loss_type huber \
    --gamma 0.8 \
    --seed 345 \
    --device $DEVICES \
    --num_workers 4 \
    || exit 1
