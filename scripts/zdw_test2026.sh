#!/bin/bash


#!/bin/bash

ITERS=200
TEST_ITERS="0 50 100 200 $ITERS" 
CONFIG=config4
QUANT_TYPE=lsq    # vanilla or lsq
MODEL_BASE="F:/3dgs_data/models"
OUTPUT_BASE="F:/3dgs_data/my_RAHT_results2026/lsq0308"
CSV_BASE="F:/3dgs_data/my_RAHT_results2026/lsq0308/csv"

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

    python mesongs.py -s "$DATAPATH" \
        --given_ply_path "$INITIALPATH" \
        --save_imp \
        --eval \
        --iterations $ITERS \
        --scene_imp $SCENE \
        --raht \
        --per_block_quant \
        --hyper_config $CONFIG \
        --csv_path "$CSVPATH" \
        --model_path "$SAVEPATH" \
        --lambda_sparsity 5e-7 \
        --quant $QUANT_TYPE \
        --test_iterations $TEST_ITERS 
        #--debug 

    echo "=== Finished scene: $SCENE ==="
    echo
}

# TUM scenes
# SCENES=("train" "truck")
SCENES=("train")
for SCENE in "${SCENES[@]}"; do
    process_scene "$SCENE" "F:/3dgs_data/image&sparse/$SCENE"
done

# db
#SCENES=("drjohnson" "playroom")
#for SCENE in "${SCENES[@]}"; do
#    process_scene "$SCENE" "F:/3dgs_data/image&sparse/$SCENE"
#done


# 360_v2 scenes
#SCENES=("counter" "room" "bicycle" "bonsai" "kitchen" "garden" "stump")
#SCENES=("room")
#for SCENE in "${SCENES[@]}"; do
#    process_scene "$SCENE" "F:/3dgs_data/image&sparse/$SCENE"
#done


