
ITERS=50
CONFIG="config4"          # 配置文件
OUTPUT_BASE="/data/zdw/zdw_data_2025/newRAHT/myMesonGS/results_2026/lsq0306"

process_scene () {
    local SCENE=$1
    local DATAPATH=$2
    echo "=== Processing scene: $SCENE ==="
    SAVEPATH="$OUTPUT_BASE/${SCENE}_${CONFIG}"
    # 调试：打印路径
    echo "CSVPATH: $CSVPATH"
    echo "SAVEPATH: $SAVEPATH"

    export CUDA_LAUNCH_BLOCKING=1
    export DEBUG_RENDER=1
    CUDA_VISIBLE_DEVICES=0 python render.py -s "$DATAPATH" -m "$SAVEPATH" --iteration $ITERS --dec_npz --eval --skip_train
    
    echo "=== Finished scene: $SCENE ==="
}

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
