#!/usr/bin/env python3
"""
独立的，可以从NPZ文件生成解压缩的PLY文件
只生成PLY
我自己加的
"""

import sys
import os
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene.gaussian_model import GaussianModel

def generate_ply(model_path, iteration):
    """
    从NPZ生成PLY文件
    
    Args:
        model_path: 模型路径
        iteration: 迭代次数
    """
    print("="*70)
    print("从NPZ生成解压缩的PLY文件")
    print("="*70)
    print(f"模型路径: {model_path}")
    print(f"迭代: {iteration}")
    print()
    
    # 构建NPZ路径
    npz_path = os.path.join(
        model_path, 
        "point_cloud", 
        f"iteration_{iteration}", 
        "pc_npz"
    )
    
    if not os.path.exists(npz_path):
        print(f"✗ 错误: NPZ路径不存在: {npz_path}")
        return False
    
    # 创建高斯模型
    print("创建高斯模型...")
    gaussians = GaussianModel(sh_degree=3)
    
    # 加载NPZ
    print("加载NPZ文件...")
    try:
        gaussians.load_npz(npz_path)
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return False
    
    # 保存PLY
    ply_path = os.path.join(
        model_path,
        "point_cloud",
        f"iteration_{iteration}",
        "decompressed.ply"
    )
    
    print("\n保存PLY文件...")
    try:
        gaussians.save_decompressed_ply(ply_path)
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        return False
    
    print("\n" + "="*70)
    print("✓ 成功！")
    print(f"PLY文件: {ply_path}")
    print("="*70)
    
    return True

def main():
    if len(sys.argv) < 3:
        print("用法: python generate_decompressed_ply.py <MODEL_PATH> <ITERATION>")
        print()
        print("示例:")
        print("  python generate_decompressed_ply.py D:/3DGS_seq/result_mesongs/output/truck_config3 20")
        print()
        print("这将生成:")
        print("  <MODEL_PATH>/point_cloud/iteration_20/decompressed.ply")
        sys.exit(1)
    
    model_path = sys.argv[1]
    iteration = sys.argv[2]
    
    if not os.path.exists(model_path):
        print(f"✗ 错误: 模型路径不存在: {model_path}")
        sys.exit(1)
    
    success = generate_ply(model_path, iteration)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
