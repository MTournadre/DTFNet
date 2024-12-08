#SEQ=/data/Datasets/movi_e/test/12
#REF_IDX=14
#OUT_FOLDER=/tmp/dtf/movi
#START=0
#END=24

#SEQ=/data/Datasets/MPI-Sintel/test/clean/mountain_2
#REF_IDX=30
#OUT_FOLDER=/tmp/dtf/mountain
#START=20
#END=50

#SEQ=/data/Datasets/MPI-Sintel/test/clean/cave_3
#REF_IDX=17
#OUT_FOLDER=/tmp/dtf/cave
#START=0
#END=30

SEQ=/data/Datasets/MPI-Sintel/test/clean/wall
REF_IDX=18
OUT_FOLDER=/tmp/dtf/wall
START=13
END=43

# Create/clean output folder
mkdir -p $OUT_FOLDER
rm -rf $OUT_FOLDER/dtfnet 2> /dev/null
rm -rf $OUT_FOLDER/mft 2> /dev/null
rm -rf $OUT_FOLDER/pips 2> /dev/null
rm -rf $OUT_FOLDER/pips2 2> /dev/null
rm -rf $OUT_FOLDER/raft-ref 2> /dev/null
rm -rf $OUT_FOLDER/raft-chain 2> /dev/null
rm -rf $OUT_FOLDER/tapir 2> /dev/null
rm -rf $OUT_FOLDER/cotracker 2> /dev/null
rm -rf $OUT_FOLDER/flowformer-ref 2> /dev/null
rm -rf $OUT_FOLDER/flowformer-chain 2> /dev/null

# DTF-Net
echo " --------------- DTF-Net ------------------"
PYTHONPATH="$(pwd)/.." python evaluate_model.py \
    --model-path models/dtfnet_v1.pt \
    --model-archi dtfnet --iters 1 --batch-size 1 \
    --downscale-max-dim 512 \
    --test-seq ${SEQ} --test-output ${OUT_FOLDER}/dtfnet \
    --test-ref-idx ${REF_IDX} --test-start $START --test-end $END
    #--resize-infer 256 256 \


echo " --------------- MFT ------------------"
PYTHONPATH="$(pwd)/../MFT:$(pwd)/.." python evaluate_model.py \
    --model-path ../MFT/configs/MFT_cfg.py \
    --model-archi mft --dtf-grid-size 1 \
    --batch-size 1 \
    --downscale-max-dim 512 \
    --test-seq ${SEQ} --test-output ${OUT_FOLDER}/mft \
    --test-ref-idx ${REF_IDX} --test-start $START --test-end $END

echo " --------------- PIPs ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../pips" python evaluate_model.py \
    --model-path ../pips/reference_model/model-000100000.pth \
    --model-archi pips --iters 3 --chain-len 8 --dtf-grid-size 2 \
    --batch-size 128 \
    --downscale-max-dim 512 \
    --test-seq ${SEQ} --test-output ${OUT_FOLDER}/pips \
    --test-ref-idx ${REF_IDX} --test-start $START --test-end $END

echo " --------------- PIPs++ ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../pips2" python evaluate_model.py \
    --model-path ../pips2/reference_model/model-000200000.pth \
    --model-archi pips2 --iters 6 --dtf-grid-size 2 \
    --batch-size 32 \
    --downscale-max-dim 512 \
    --test-seq ${SEQ} --test-output ${OUT_FOLDER}/pips2 \
    --test-ref-idx ${REF_IDX} --test-start $START --test-end $END


echo " --------------- RAFT ref ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../RAFT/core" python evaluate_model.py \
    --model-path ../RAFT/models/raft-things.pth \
    --model-archi raft --iters 12 \
    --batch-size 1 \
    --downscale-max-dim 512 \
    --test-seq ${SEQ} --test-output ${OUT_FOLDER}/raft-ref \
    --test-ref-idx ${REF_IDX} --test-start $START --test-end $END

echo " --------------- RAFT chain ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../RAFT/core" python evaluate_model.py \
    --model-path ../RAFT/models/raft-things.pth \
    --model-archi raft --iters 12 --chain-len 2 \
    --batch-size 1 \
    --downscale-max-dim 512 \
    --test-seq ${SEQ} --test-output ${OUT_FOLDER}/raft-chain \
    --test-ref-idx ${REF_IDX} --test-start $START --test-end $END

echo " --------------- FLOWFORMER ref ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../FlowFormer:$(pwd)/../FlowFormer/core" python evaluate_model.py \
    --model-path ../FlowFormer/checkpoints/sintel.pth \
    --model-archi flowformer --iters 12 \
    --batch-size 1 \
    --downscale-max-dim 512 \
    --test-seq ${SEQ} --test-output ${OUT_FOLDER}/flowformer-ref \
    --test-ref-idx ${REF_IDX} --test-start $START --test-end $END

echo " --------------- FLOWFORMER chain ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../FlowFormer:$(pwd)/../FlowFormer/core" python evaluate_model.py \
    --model-path ../FlowFormer/checkpoints/sintel.pth \
    --model-archi flowformer --iters 12 --chain-len 2 \
    --batch-size 1 \
    --downscale-max-dim 512 \
    --test-seq ${SEQ} --test-output ${OUT_FOLDER}/flowformer-chain \
    --test-ref-idx ${REF_IDX} --test-start $START --test-end $END

echo " --------------- TAPIR ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../tapnet" python evaluate_model.py \
    --model-path ../tapnet/checkpoints/bootstapir_checkpoint_v2.pt \
    --model-archi tapir --iters 3 --chain-len 24 --dtf-grid-size 2 \
    --resize-infer 256 256 \
    --batch-size 128 \
    --downscale-max-dim 512 \
    --test-seq ${SEQ} --test-output ${OUT_FOLDER}/tapir \
    --test-ref-idx ${REF_IDX} --test-start $START --test-end $END

echo " --------------- CoTracker ------------------"
PYTHONPATH="$(pwd)/..:$(pwd)/../co-tracker" python evaluate_model.py \
    --model-path ../co-tracker/checkpoints/cotracker_stride_4_wind_8.pth \
    --model-archi cotracker \
    --downscale-max-dim 512 \
    --test-seq ${SEQ} --test-output ${OUT_FOLDER}/cotracker \
    --test-ref-idx ${REF_IDX} --test-start $START --test-end $END



echo " --------------- Composing images (VS optical-flow) ------------------"
rm -rf $OUT_FOLDER/res 2> /dev/null
python compose_images.py \
    --sequence_path $SEQ --output_path $OUT_FOLDER/res \
    --ref_idx $REF_IDX --start_idx $START \
    --method_names \
        "FlowFormer_ref" \
        "FlowFormer_chain" \
        "MFT" \
        "DTF-Net (ours)" \
    --method_paths \
        ${OUT_FOLDER}/flowformer-ref \
        ${OUT_FOLDER}/flowformer-chain \
        ${OUT_FOLDER}/mft \
        ${OUT_FOLDER}/dtfnet \
    || exit 1
# -crf is not supported in some versions of ffmpeg
ffmpeg \
    -framerate 12 \
    -i $OUT_FOLDER/res/frame_%04d.png \
    -filter_complex "[0]setpts=4*PTS[2];[0][2][0]concat=n=3" \
    -y -pix_fmt yuv420p -b:v 8600k $OUT_FOLDER/dtf_vs_of.mp4


echo " --------------- Composing images (VS trajectory) ------------------"
rm -rf $OUT_FOLDER/res 2> /dev/null
python compose_images.py \
    --sequence_path $SEQ --output_path $OUT_FOLDER/res \
    --ref_idx $REF_IDX --start_idx $START \
    --method_names \
        "PIPs++" \
        "TAPIR" \
        "CoTracker" \
        "DTF-Net (ours)" \
    --method_paths \
        ${OUT_FOLDER}/pips2 \
        ${OUT_FOLDER}/tapir \
        ${OUT_FOLDER}/cotracker \
        ${OUT_FOLDER}/dtfnet \
    || exit 1
# -crf is not supported in some versions of ffmpeg
ffmpeg \
    -framerate 12 \
    -i $OUT_FOLDER/res/frame_%04d.png \
    -filter_complex "[0]setpts=4*PTS[2];[0][2][0]concat=n=3" \
    -y -pix_fmt yuv420p -b:v 8600k $OUT_FOLDER/dtf_vs_traj.mp4
