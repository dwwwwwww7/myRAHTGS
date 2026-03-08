#!/bin/bash

# ============================================================================
# MesonGS 批量训练、渲染和评估脚本
# ============================================================================
# 功能：
# 1. 对多个场景进行 RAHT 压缩感知微调（可选）
# 2. 渲染测试集和训练集
# 3. 计算质量指标（PSNR, SSIM, LPIPS）
# 
# 使用方法：
# - 设置 SKIP_TRAINING=true  : 跳过训练，使用已有模型
# - 设置 SKIP_TRAINING=false : 执行完整训练流程
# ============================================================================

# ============================================================================
# 配置参数
# ============================================================================

# 数据集路径
DATA_BASE="E:/3dgs data/image&sparse"
MODEL_BASE="E:/3dgs data/models"
OUTPUT_BASE="E:/3dgs data/MesonGS/output"
CSV_BASE="E:/3dgs data/MesonGS/exp_data"
mkdir -p "$OUTPUT_BASE"
mkdir -p "$CSV_BASE"

# 要处理的场景列表
declare -a SCENES=(
    "playroom"
	#"bicycle"
    # "bonsai"
    # "counter"
    # "kitchen"
    # "room"
    # "stump"
    # "garden"
    # "train"
     #"truck"
    # "chair"
    # "drums"
    # "ficus"
    # "hotdog"
    # "lego"
    # "mic"
    # "materials"
    # "ship"
)

# 训练参数
ITERS=0                    # 微调迭代次数
CONFIG="config3"            # 配置文件
TEST_ITERS="0 $ITERS"       # 测试迭代（0=初始评估，ITERS=最终评估）

# 流程控制
SKIP_TRAINING=false          # 是否跳过训练步骤（使用已有模型）
SKIP_TRAIN=true             # 是否跳过训练集渲染
SKIP_TEST=false             # 是否跳过测试集渲染

# GPU 设备
CUDA_DEVICE=0

# 日志目录
LOG_DIR="logs_batch"
mkdir -p "$LOG_DIR"

# ============================================================================
# 辅助函数
# ============================================================================

# 打印分隔线
print_separator() {
    echo ""
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
    echo ""
}

# 打印子标题
print_subtitle() {
    echo ""
    echo "---------- $1 ----------"
    echo ""
}

# 检查命令是否成功
check_status() {
    if [ $1 -ne 0 ]; then
        echo "错误: $2 失败，退出码: $1"
        return 1
    else
        echo "✓ $2 成功"
        return 0
    fi
}

# ============================================================================
# 主循环
# ============================================================================

print_separator "MesonGS 批量训练、渲染和评估"
echo "场景列表: ${SCENES[@]}"
echo "微调迭代: $ITERS"
echo "配置文件: $CONFIG"
echo "GPU 设备: $CUDA_DEVICE"
print_separator "开始处理"

# 记录开始时间
START_TIME=$(date +%s)

# 统计信息
TOTAL_SCENES=${#SCENES[@]}
SUCCESS_COUNT=0
FAIL_COUNT=0

for SCENE in "${SCENES[@]}"; do
    print_separator "处理场景: $SCENE"
    
    # 构建路径
    DATAPATH="$DATA_BASE/$SCENE"
    INITIALPATH="$MODEL_BASE/$SCENE/point_cloud/iteration_30000/point_cloud.ply"
    CSVPATH="$CSV_BASE/${SCENE}_${CONFIG}.csv"
    SAVEPATH="$OUTPUT_BASE/${SCENE}_${CONFIG}"
    
    # 日志文件
    TRAIN_LOG="$LOG_DIR/${SCENE}_train.log"
    RENDER_LOG="$LOG_DIR/${SCENE}_render.log"
    METRICS_LOG="$LOG_DIR/${SCENE}_metrics.log"
    
    # ========================================================================
    # 步骤 1: 检查预训练模型是否存在
    # ========================================================================
    print_subtitle "步骤 1/4: 检查预训练模型"
    
    if [ ! -f "$INITIALPATH" ]; then
        echo "错误: 找不到预训练模型: $INITIALPATH"
        echo "跳过场景 $SCENE"
        ((FAIL_COUNT++))
        continue
    fi
    echo "✓ 预训练模型存在: $INITIALPATH"
    
    if [ ! -d "$DATAPATH" ]; then
        echo "错误: 找不到数据集: $DATAPATH"
        echo "跳过场景 $SCENE"
        ((FAIL_COUNT++))
        continue
    fi
    echo "✓ 数据集存在: $DATAPATH"
    
    # ========================================================================
    # 步骤 2: 微调训练
    # ========================================================================
    print_subtitle "步骤 2/4: RAHT 压缩感知微调"
    
    if [ "$SKIP_TRAINING" = true ]; then
        # 跳过训练 - 使用已有模型
        echo "跳过训练步骤 - 使用已有模型"
        echo "模型路径: $SAVEPATH"
        echo "目标迭代: $ITERS"
        
        # 检查模型目录是否存在
        MODEL_PATH="$SAVEPATH/point_cloud/iteration_$ITERS"
        if [ ! -d "$MODEL_PATH" ]; then
            echo "错误: 找不到模型目录: $MODEL_PATH"
            ((FAIL_COUNT++))
            continue
        fi
        
        # 检查压缩文件是否存在
        NPZ_PATH="$MODEL_PATH/pc_npz/bins.zip"
        if [ ! -f "$NPZ_PATH" ]; then
            echo "错误: 找不到压缩文件: $NPZ_PATH"
            ((FAIL_COUNT++))
            continue
        fi
        
        echo "✓ 找到已训练模型"
        echo "  模型目录: $MODEL_PATH"
        echo "  压缩文件: $NPZ_PATH"
        echo "  文件大小: $(du -h "$NPZ_PATH" | cut -f1)"
        
        TRAIN_STATUS=0  # 标记为成功
    else
        # 执行训练
        echo "开始微调训练..."
        echo "日志文件: $TRAIN_LOG"
        
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python mesongs.py \
            -s "$DATAPATH" \
            -m "$SAVEPATH" \
            --given_ply_path "$INITIALPATH" \
            --hyper_config $CONFIG \
            --csv_path "$CSVPATH" \
            --scene_imp $SCENE \
            --iterations $ITERS \
            --test_iterations $TEST_ITERS \
            --finetune_lr_scale 1 \
            --raht \
            --per_block_quant \
            --clamp_color \
            --convert_SHs_python \
            --save_imp \
            --eval \
            2>&1 | tee "$TRAIN_LOG"
        
        TRAIN_STATUS=$?
    fi
    
    if ! check_status $TRAIN_STATUS "微调训练/模型检查"; then
        echo "跳过后续步骤"
        ((FAIL_COUNT++))
        continue
    fi
    
    # ========================================================================
    # 步骤 3: 渲染
    # ========================================================================
    print_subtitle "步骤 3/4: 渲染测试集"
    
    echo "开始渲染..."
    echo "日志文件: $RENDER_LOG"
    
    # 构建渲染命令（参考 render.sh）
    RENDER_CMD="python render.py -s \"$DATAPATH\" -m \"$SAVEPATH\" --iteration $ITERS --dec_npz --eval"
    
    # 添加场景名称和日志名称
    RENDER_CMD="$RENDER_CMD --scene_name $SCENE --log_name ${SCENE}_${CONFIG}"
    
    # 添加保存目录名称（用于 metrics.py）
    RENDER_CMD="$RENDER_CMD --save_dir_name ours"
    
    if [ "$SKIP_TRAIN" = true ]; then
        RENDER_CMD="$RENDER_CMD --skip_train"
    fi
    
    if [ "$SKIP_TEST" = true ]; then
        RENDER_CMD="$RENDER_CMD --skip_test"
    fi
    
    echo "命令: $RENDER_CMD"
    
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE eval $RENDER_CMD 2>&1 | tee "$RENDER_LOG"
    
    RENDER_STATUS=$?
    
    if ! check_status $RENDER_STATUS "渲染"; then
        echo "跳过后续步骤"
        ((FAIL_COUNT++))
        continue
    fi
    
    # ========================================================================
    # 步骤 4: 计算质量指标
    # ========================================================================
    print_subtitle "步骤 4/4: 计算质量指标"
    
    echo "计算 PSNR, SSIM, LPIPS..."
    echo "日志文件: $METRICS_LOG"
    
    python metrics.py -m "$SAVEPATH" 2>&1 | tee "$METRICS_LOG"
    
    METRICS_STATUS=$?
    
    if ! check_status $METRICS_STATUS "质量指标计算"; then
        ((FAIL_COUNT++))
        continue
    fi
    
    # ========================================================================
    # 显示结果
    # ========================================================================
    print_subtitle "场景 $SCENE 处理完成"
    
    # 显示 results.json
    RESULTS_FILE="$SAVEPATH/results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "质量指标:"
        cat "$RESULTS_FILE"
    fi
    
    ((SUCCESS_COUNT++))
    
    print_separator "场景 $SCENE 全部完成"
    
done

# ============================================================================
# 最终总结
# ============================================================================

# 计算总耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

print_separator "批量处理完成"

echo "处理摘要:"
echo "  总场景数: $TOTAL_SCENES"
echo "  成功: $SUCCESS_COUNT"
echo "  失败: $FAIL_COUNT"
echo "  总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

echo "详细结果:"
for SCENE in "${SCENES[@]}"; do
    SAVEPATH="$OUTPUT_BASE/${SCENE}_${CONFIG}"
    RESULTS_FILE="$SAVEPATH/results.json"
    
    echo ""
    echo "  场景: $SCENE"
    echo "  输出路径: $SAVEPATH"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "  ✓ 质量指标已计算"
        # 提取关键指标（如果有 jq 工具）
        if command -v jq &> /dev/null; then
            PSNR=$(jq -r '.ours_'$ITERS'.PSNR' "$RESULTS_FILE" 2>/dev/null)
            SSIM=$(jq -r '.ours_'$ITERS'.SSIM' "$RESULTS_FILE" 2>/dev/null)
            LPIPS=$(jq -r '.ours_'$ITERS'.LPIPS' "$RESULTS_FILE" 2>/dev/null)
            
            if [ "$PSNR" != "null" ]; then
                echo "    PSNR:  $PSNR dB"
                echo "    SSIM:  $SSIM"
                echo "    LPIPS: $LPIPS"
            fi
        fi
    else
        echo "  ✗ 处理失败或未完成"
    fi
    
    # 检查压缩文件大小
    NPZ_PATH="$SAVEPATH/point_cloud/iteration_$ITERS/pc_npz/bins.zip"
    if [ -f "$NPZ_PATH" ]; then
        SIZE=$(du -h "$NPZ_PATH" | cut -f1)
        echo "  压缩文件大小: $SIZE"
    fi
done

echo ""
print_separator "全部完成"

# ============================================================================
# 生成汇总 CSV
# ============================================================================

SUMMARY_CSV="$CSV_BASE/batch_summary_${CONFIG}.csv"
echo "生成汇总 CSV: $SUMMARY_CSV"

# CSV 表头
echo "Scene,PSNR,SSIM,LPIPS,Size_MB,Status" > "$SUMMARY_CSV"

for SCENE in "${SCENES[@]}"; do
    SAVEPATH="$OUTPUT_BASE/${SCENE}_${CONFIG}"
    RESULTS_FILE="$SAVEPATH/results.json"
    NPZ_PATH="$SAVEPATH/point_cloud/iteration_$ITERS/pc_npz/bins.zip"
    
    if [ -f "$RESULTS_FILE" ] && command -v jq &> /dev/null; then
        PSNR=$(jq -r '.ours_'$ITERS'.PSNR' "$RESULTS_FILE" 2>/dev/null)
        SSIM=$(jq -r '.ours_'$ITERS'.SSIM' "$RESULTS_FILE" 2>/dev/null)
        LPIPS=$(jq -r '.ours_'$ITERS'.LPIPS' "$RESULTS_FILE" 2>/dev/null)
        
        if [ -f "$NPZ_PATH" ]; then
            SIZE_BYTES=$(stat -f%z "$NPZ_PATH" 2>/dev/null || stat -c%s "$NPZ_PATH" 2>/dev/null)
            SIZE_MB=$(echo "scale=2; $SIZE_BYTES / 1024 / 1024" | bc)
        else
            SIZE_MB="N/A"
        fi
        
        if [ "$PSNR" != "null" ]; then
            echo "$SCENE,$PSNR,$SSIM,$LPIPS,$SIZE_MB,Success" >> "$SUMMARY_CSV"
        else
            echo "$SCENE,N/A,N/A,N/A,$SIZE_MB,Failed" >> "$SUMMARY_CSV"
        fi
    else
        echo "$SCENE,N/A,N/A,N/A,N/A,Failed" >> "$SUMMARY_CSV"
    fi
done

echo "✓ 汇总 CSV 已生成"
echo ""

# 显示汇总表格
if command -v column &> /dev/null; then
    echo "汇总表格:"
    column -t -s, "$SUMMARY_CSV"
else
    cat "$SUMMARY_CSV"
fi

echo ""
echo "所有日志文件保存在: $LOG_DIR"
echo "所有结果保存在: $OUTPUT_BASE"
echo ""
