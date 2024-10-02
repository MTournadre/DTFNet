DATASET_ROOT=/data

echo " --------------- MFT ------------------"
PYTHONPATH="$(pwd)/../MFT:$(pwd)/.." python evaluate_model.py \
    --model-path ../MFT/configs/MFT_cfg.py \
    --model-archi mft --chain-len 250 --dtf-grid-size 1 \
    --davis-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_davis/tapvid_davis.pkl \
    --rgb-stacking-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
    --query-mode first

PYTHONPATH="$(pwd)/../MFT:$(pwd)/.." python evaluate_model.py \
    --model-path ../MFT/configs/MFT_cfg.py \
    --model-archi mft --chain-len 250 --dtf-grid-size 1 \
    --downscale-max-dim 512 \
    --crop-window-size 8 --crop-strategy future \
    --sintel-dir $DATASET_ROOT/optflow/MPI-Sintel/validation \
    --kubric-dtf-dir-path $DATASET_ROOT/traj

echo " --------------- PIPs (first) ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../pips" python evaluate_model.py \
    --model-path ../pips/reference_model/model-000100000.pth \
    --model-archi pips --iters 6 --chain-len 8 \
    --batch-size 1024 \
    --davis-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_davis/tapvid_davis.pkl \
    --rgb-stacking-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
    --sintel-dir $DATASET_ROOT/optflow/MPI-Sintel/validation \
    --query-mode first

PYTHONPATH="$(pwd)/..:$(pwd)/../pips" python evaluate_model.py \
    --model-path ../pips/reference_model/model-000100000.pth \
    --model-archi pips --iters 6 --chain-len 8 --dtf-grid-size 4 \
    --crop-window-size 8 --crop-strategy future \
    --batch-size 1024 \
    --downscale-max-dim 512 \
    --kubric-dtf-dir-path $DATASET_ROOT/traj


echo " --------------- PIPs++ (first) ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../pips2" python evaluate_model.py \
    --model-path ../pips2/reference_model/model-000200000.pth \
    --model-archi pips2 --iters 6 --chain-len 128 --dtf-grid-size 4 \
    --crop-window-size 8 --crop-strategy future \
    --batch-size 1024 \
    --downscale-max-dim 512 \
    --davis-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_davis/tapvid_davis.pkl \
    --rgb-stacking-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
    --sintel-dir $DATASET_ROOT/optflow/MPI-Sintel/validation \
    --kubric-dtf-dir-path $DATASET_ROOT/traj \
    --query-mode first

echo " --------------- TAPIR (first) ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../tapnet" python evaluate_model.py \
    --model-path ../tapnet/checkpoints/bootstapir_checkpoint_v2.pt \
    --model-archi tapir --iters 6 --dtf-grid-size 4 \
    --crop-window-size 8 --crop-strategy future \
    --batch-size 1024 \
    --downscale-max-dim 512 \
    --resize-inffer 256 256 \
    --davis-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_davis/tapvid_davis.pkl \
    --rgb-stacking-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
    --sintel-dir $DATASET_ROOT/optflow/MPI-Sintel/validation \
    --kubric-dtf-dir-path $DATASET_ROOT/traj \
    --query-mode first

echo " --------------- COTRACKER (first) ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../co-tracker" python evaluate_model.py \
    --model-path ../co-tracker/checkpoints/cotracker_stride_4_wind_8.pth \
    --model-archi cotracker --dtf-grid-size 8 \
    --crop-window-size 8 --crop-strategy centered \
    --batch-size 1024 \
    --downscale-max-dim 512 \
    --davis-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_davis/tapvid_davis.pkl \
    --rgb-stacking-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
    --sintel-dir $DATASET_ROOT/optflow/MPI-Sintel/validation \
    --kubric-dtf-dir-path $DATASET_ROOT/traj \
    --query-mode first


echo " --------------- RAFT ref (first) ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../RAFT/core" python evaluate_model.py \
    --model-path ../RAFT/models/raft-things.pth \
    --model-archi raft --iters 12 --chain-len 250 \
    --batch-size 24 \
    --crop-window-size 4 --crop-strategy centered \
    --batch-size 24 \
    --downscale-max-dim 512 \
    --davis-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_davis/tapvid_davis.pkl \
    --rgb-stacking-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
    --sintel-dir $DATASET_ROOT/optflow/MPI-Sintel/validation \
    --kubric-dtf-dir-path $DATASET_ROOT/traj \
    --query-mode first

# RAFT_chain
echo " --------------- RAFT chain (first) ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../RAFT/core" python evaluate_model.py \
    --model-path ../RAFT/models/raft-things.pth \
    --model-archi raft --iters 12 --chain-len 2 \
    --batch-size 24 \
    --crop-window-size 4 --crop-strategy centered \
    --batch-size 24 \
    --downscale-max-dim 512 \
    --davis-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_davis/tapvid_davis.pkl \
    --rgb-stacking-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
    --sintel-dir $DATASET_ROOT/optflow/MPI-Sintel/validation \
    --kubric-dtf-dir-path $DATASET_ROOT/traj \
    --query-mode first

# FlowFormer_chain
echo " --------------- FlowFormer ref (first) ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../FlowFormer:$(pwd)/../FlowFormer/core" python evaluate_model.py \
    --model-path ../FlowFormer/checkpoints/sintel.pth \
    --model-archi flowformer --iters 12 \
    --batch-size 24 \
    --crop-window-size 4 --crop-strategy centered \
    --downscale-max-dim 512 \
    --davis-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_davis/tapvid_davis.pkl \
    --rgb-stacking-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
    --sintel-dir $DATASET_ROOT/optflow/MPI-Sintel/validation \
    --kubric-dtf-dir-path $DATASET_ROOT/traj

# FlowFormer_ref
echo " --------------- FlowFormer chain (first) ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../FlowFormer:$(pwd)/../FlowFormer/core" python evaluate_model.py \
    --model-path ../FlowFormer/checkpoints/sintel.pth \
    --model-archi flowformer --iters 12 --chain-len 2 \
    --batch-size 24 \
    --crop-window-size 4 --crop-strategy centered \
    --downscale-max-dim 512 \
    --davis-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_davis/tapvid_davis.pkl \
    --rgb-stacking-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
    --sintel-dir $DATASET_ROOT/optflow/MPI-Sintel/validation \
    --kubric-dtf-dir-path $DATASET_ROOT/traj



echo " --------------- DTF: DTF-Net ------------------"
MODEL_PATH=models/dtfnet_final.pt
PYTHONPATH="$(pwd)/.." python evaluate_model.py \
    --model-path ${MODEL_PATH} \
    --model-archi dtfnet --iters 1 --batch-size 1 \
    --kubric-dtf-dir-path $DATASET_ROOT/traj

echo "optflow with 4 surrounding frames..."
PYTHONPATH="$(pwd)/.." python evaluate_model.py \
    --model-path ${MODEL_PATH} \
    --model-archi dtfnet --iters 1 --batch-size 1 \
    --crop-window-size 4 --crop-strategy centered \
    --downscale-max-dim 512 \
    --resize-infer 256 256 \
    --sintel-dir $DATASET_ROOT/optflow/MPI-Sintel/validation

echo "optflow with 8 surrounding frames..."
PYTHONPATH="$(pwd)/.." python evaluate_model.py \
    --model-path ${MODEL_PATH} \
    --model-archi dtfnet --iters 1 --batch-size 1 \
    --crop-window-size 8 --crop-strategy centered \
    --downscale-max-dim 512 \
    --resize-infer 256 256 \
    --sintel-dir $DATASET_ROOT/optflow/MPI-Sintel/validation

echo "optflow with 12 surrounding frames..."
PYTHONPATH="$(pwd)/.." python evaluate_model.py \
    --model-path ${MODEL_PATH} \
    --model-archi dtfnet --iters 1 --batch-size 1 \
    --crop-window-size 12 --crop-strategy centered \
    --downscale-max-dim 512 \
    --resize-infer 256 256 \
    --sintel-dir $DATASET_ROOT/optflow/MPI-Sintel/validation

echo "optflow with 16 surrounding frames..."
PYTHONPATH="$(pwd)/.." python evaluate_model.py \
    --model-path ${MODEL_PATH} \
    --model-archi dtfnet --iters 1 --batch-size 1 \
    --crop-window-size 16 --crop-strategy centered \
    --downscale-max-dim 512 \
    --resize-infer 256 256 \
    --sintel-dir $DATASET_ROOT/optflow/MPI-Sintel/validation

echo "Trajectories..."
PYTHONPATH="$(pwd)/.." python evaluate_model.py \
    --model-path ${MODEL_PATH} \
    --model-archi dtfnet --iters 1 --batch-size 1 --chain-len 150 \
    --downscale-max-dim 512 \
    --query-mode first \
    --rgb-stacking-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl
    --davis-pkl-path $DATASET_ROOT/TAP-Vid/tapvid_davis/tapvid_davis.pkl

