#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, 
               save_images=True, compute_metrics=True):
    """
    渲染指定视角集合并计算质量指标
    
    Args:
        model_path: 模型路径
        name: 数据集名称 (train/test)
        iteration: 迭代次数
        views: 相机视角列表
        gaussians: 高斯模型
        pipeline: 渲染管线参数
        background: 背景颜色
        save_images: 是否保存渲染图像
        compute_metrics: 是否计算质量指标
    """
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    if save_images:
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

    psnrs = []
    ssims = []
    lpipss = []

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        # 渲染图像
        rendering = render(view, gaussians, pipeline, background, 
                          debug=False, clamp_color=True)["render"]
        gt = view.original_image[0:3, :, :]

        # 保存图像
        if save_images:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # 计算质量指标
        if compute_metrics:
            psnrs.append(psnr(rendering, gt))
            ssims.append(ssim(rendering, gt))
            lpipss.append(lpips(rendering, gt, net_type='vgg'))

    # 计算平均指标
    if compute_metrics and len(psnrs) > 0:
        psnr_val = torch.stack(psnrs).mean()
        ssim_val = torch.tensor(ssims).mean()
        lpips_val = torch.tensor(lpipss).mean()
        
        print(f"\n{name.upper()} Set Metrics:")
        print(f"  PSNR:  {psnr_val:.4f} dB")
        print(f"  SSIM:  {ssim_val:.4f}")
        print(f"  LPIPS: {lpips_val:.4f}")
        
        return psnr_val.item(), ssim_val.item(), lpips_val.item()
    
    return None, None, None


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, 
                skip_train: bool, skip_test: bool, given_ply_path: str = None,
                save_images: bool = True, compute_metrics: bool = True):
    """
    渲染训练集和测试集
    
    Args:
        dataset: 数据集参数
        iteration: 加载的迭代次数
        pipeline: 渲染管线参数
        skip_train: 是否跳过训练集
        skip_test: 是否跳过测试集
        given_ply_path: 指定的 PLY 文件路径（如果提供，将从此文件加载）
        save_images: 是否保存渲染图像
        compute_metrics: 是否计算质量指标
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        
        # 加载场景和模型
        scene = Scene(dataset, gaussians, load_iteration=iteration, 
                     shuffle=False, given_ply_path=given_ply_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        print(f"\n{'='*70}")
        print(f"渲染配置:")
        print(f"  模型路径: {dataset.model_path}")
        if given_ply_path:
            print(f"  PLY 文件: {given_ply_path}")
        print(f"  迭代次数: {scene.loaded_iter}")
        print(f"  高斯点数: {gaussians.get_xyz.shape[0]:,}")
        print(f"  SH 阶数: {gaussians.max_sh_degree}")
        print(f"  背景颜色: {bg_color}")
        print(f"{'='*70}\n")

        results = {}

        if not skip_train:
            train_metrics = render_set(dataset.model_path, "train", scene.loaded_iter, 
                                      scene.getTrainCameras(), gaussians, pipeline, 
                                      background, save_images, compute_metrics)
            results['train'] = train_metrics

        if not skip_test:
            test_metrics = render_set(dataset.model_path, "test", scene.loaded_iter, 
                                     scene.getTestCameras(), gaussians, pipeline, 
                                     background, save_images, compute_metrics)
            results['test'] = test_metrics

        return results


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Render script for 3D Gaussian Splatting PLY files")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    # 基本参数
    parser.add_argument("--iteration", default=-1, type=int, 
                       help="加载的迭代次数，-1 表示最新")
    parser.add_argument("--skip_train", action="store_true", 
                       help="跳过训练集渲染")
    parser.add_argument("--skip_test", action="store_true", 
                       help="跳过测试集渲染")
    parser.add_argument("--quiet", action="store_true", 
                       help="静默模式")
    
    # 新增参数：支持指定 PLY 文件
    parser.add_argument("--ply_path", type=str, default=None,
                       help="指定要渲染的 PLY 文件路径（可选）")
    parser.add_argument("--no_save", action="store_true",
                       help="不保存渲染图像，仅计算指标")
    parser.add_argument("--no_metrics", action="store_true",
                       help="不计算质量指标，仅保存图像")
    
    args = get_combined_args(parser)
    
    # 安全获取新增参数（兼容从配置文件加载的情况）
    ply_path = getattr(args, 'ply_path', None)
    no_save = getattr(args, 'no_save', False)
    no_metrics = getattr(args, 'no_metrics', False)
    
    print("\n" + "="*70)
    print("3D Gaussian Splatting - PLY 渲染工具")
    print("="*70)
    print(f"模型路径: {args.model_path}")
    if ply_path:
        print(f"PLY 文件: {ply_path}")
    print("="*70 + "\n")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # 执行渲染
    results = render_sets(
        model.extract(args), 
        args.iteration, 
        pipeline.extract(args), 
        args.skip_train, 
        args.skip_test,
        given_ply_path=ply_path,
        save_images=not no_save,
        compute_metrics=not no_metrics
    )
    
    print("\n" + "="*70)
    print("渲染完成！")
    print("="*70)