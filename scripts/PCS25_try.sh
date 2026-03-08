#!/bin/bash

ITERS=10000
TEST_ITERS="0 50 100 200 300 400 500 600 700 800 900 1000 2000 5000 $ITERS" 
CONFIG=config4
LSEG=0
CB=0
DEPTH=0
quantizer="lsq"  #"lsq" 或 "vanilla"

# 数据集路径 (Linux路径)
MODEL_BASE="/data/zdw/datasets/inria3DGS_pretrained_models"
OUTPUT_BASE="/data/zdw/zdw_data_2025/newRAHT/myMesonGS/results_2026/lsq0306_100001"
CSV_BASE="/data/zdw/zdw_data_2025/newRAHT/myMesonGS/results_2026/lsq0306_100001/csv"

mkdir -p "$OUTPUT_BASE"
mkdir -p "$CSV_BASE"

process_scene () {
    local SCENE=$1
    local DATAPATH=$2

    echo "=== Processing scene: $SCENE ==="
    
    MODEL_PATH="$MODEL_BASE/$SCENE"
    INITIALPATH="$MODEL_PATH/point_cloud/iteration_30000/point_cloud.ply"
    CSVPATH="$CSV_BASE/${SCENE}_${CONFIG}.csv"
    SAVEPATH="$OUTPUT_BASE/${SCENE}_${CONFIG}"
    # 调试：打印路径
    echo "CSVPATH: $CSVPATH"
    echo "SAVEPATH: $SAVEPATH"
    CUDA_VISIBLE_DEVICES=0 python mesongs.py -s "$DATAPATH" \
        --given_ply_path "$INITIALPATH" \
        --num_bits 8 \
        --save_imp \
        --eval \
        --iterations $ITERS \
        --finetune_lr_scale 1 \
        --convert_SHs_python \
        --percent 0 \
        --steps 1000 \
        --scene_imp $SCENE \
        --depth $DEPTH \
        --raht \
        --clamp_color \
        --per_block_quant \
        --lseg $LSEG \
        --debug \
        --hyper_config $CONFIG \
        --csv_path "$CSVPATH" \
        --model_path "$SAVEPATH" \
        --test_iterations $TEST_ITERS \
        --quant_type $quantizer 

    echo "=== Finished scene: $SCENE ==="
    echo
}

# 根据不同数据集注释/取消注释相应的部分

# mic scene
#SCENES=("mic" "lego" "drums" "ficus" "hotdog" "materials" "ship" "chair")
#for SCENE in "${SCENES[@]}"; do
 #   process_scene "$SCENE" "/data/zdw/datasets/nerf_synthetic/$SCENE"
#done


# TUM scenes
# SCENES=("train" "truck")
SCENES=("train")
for SCENE in "${SCENES[@]}"; do
    process_scene "$SCENE" "/data/zdw/datasets/tandt_db/tandt/$SCENE"
done

# db
#SCENES=("drjohnson" "playroom")
#for SCENE in "${SCENES[@]}"; do
#    process_scene "$SCENE" "/data/zdw/datasets/tandt_db/db/$SCENE"
#done


# 360_v2 scenes
#SCENES=("counter" "room" "bicycle" "bonsai" "kitchen" "garden" "stump")
#SCENES=("room")
#for SCENE in "${SCENES[@]}"; do
#    process_scene "$SCENE" "/data/zdw/datasets/360_v2/$SCENE"
#done
