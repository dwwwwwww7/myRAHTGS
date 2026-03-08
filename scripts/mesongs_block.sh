#!/bin/bash

# ============================================================================
# MesonGS 微调脚本 (RAHT + 分块量化)
# ============================================================================
# 说明：
# - 此脚本用于对预训练的 3DGS 模型进行 RAHT 压缩感知微调
# - 使用 config3 配置文件自动设置场景特定参数（depth, prune, n_block）
# - VQ 相关参数已移除（使用 RAHT 替代）
# ============================================================================

# ============================================================================
# 基本配置
# ============================================================================
SCENE=playroom                          # 场景名称
ITERS=0                                # 微调迭代次数
CONFIG=config3                          # 配置文件（自动设置 depth, prune, n_block）

# ============================================================================
# 路径配置
# ============================================================================
DATAPATH="E:/3dgs data/image&sparse/${SCENE}"                              # 数据集路径
INITIALPATH="E:/3dgs data/models/playroom/point_cloud/iteration_30000/point_cloud.ply"  # 预训练模型
CSVPATH="E:/3dgs data/MesonGS/exp_data/csv/${SCENE}_${CONFIG}.csv"        # CSV 日志
SAVEPATH="E:/3dgs data/MesonGS/output/${SCENE}_${CONFIG}"                 # 输出路径

# ============================================================================
# 运行微调
# ============================================================================
CUDA_VISIBLE_DEVICES=0 python mesongs.py \
    -s "$DATAPATH" \
    -m "$SAVEPATH" \
    --given_ply_path "$INITIALPATH" \
    --hyper_config $CONFIG \
    --csv_path "$CSVPATH" \
    --scene_imp $SCENE \
    --iterations $ITERS \
    --test_iterations 0 $ITERS \
    --raht \
    --per_block_quant \
    --clamp_color \
    --convert_SHs_python \
    --save_imp \
    --eval \
    --debug

# ============================================================================
# 参数说明
# ============================================================================
# 必需参数：
#   -s, --source_path       数据集路径（包含 images 和 sparse）
#   -m, --model_path        输出模型路径
#   --given_ply_path        预训练模型 PLY 文件路径
#   --hyper_config          配置文件名（config3 会自动设置 depth/prune/n_block）
#   --scene_imp             场景名称（用于从配置文件读取参数）
#
# 核心参数：
#   --iterations            微调迭代次数（默认：25）
#   --test_iterations       在哪些迭代进行测试（0 表示初始评估）
#   --raht                  启用 RAHT 变换
#   --per_block_quant       启用分块量化
#   --clamp_color           启用颜色截断（防止负值）
#
# 辅助参数：
#   --convert_SHs_python    使用 Python 转换球谐系数
#   --save_imp              保存重要性分数
#   --eval                  启用评估模式
#   --debug                 启用调试输出（仅第一张图片）
#   --csv_path              CSV 日志文件路径
#
# 自动配置参数（通过 config3 设置，无需手动指定）：
#   --depth                 八叉树深度（playroom: 20）
#   --percent               剪枝比例（playroom: 0.2 = 20%）
#   --n_block               分块数量（playroom: 57）
#   --codebook_size         码本大小（playroom: 2048，RAHT 模式下不使用）
#
# 已移除的参数（不再需要）：
#   --lseg                  （已废弃，使用 hyper_config 替代）
#   --steps                 （VQ 训练步数，RAHT 模式下不需要）
#   --batch_size            （VQ 批次大小，RAHT 模式下不需要）
#   --finetune_lr_scale     （学习率缩放，代码内部自动设置）
# ============================================================================