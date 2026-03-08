#解压缩并渲染生成的npz文件,过程中会保存解压得到的ply文件
ITERS=0
CONFIG="config4"          # 配置文件
OUTPUT_BASE="F:/3dgs_data/my_RAHT_results2026/lsq0308"

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

