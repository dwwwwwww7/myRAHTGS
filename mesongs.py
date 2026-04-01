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

import csv
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import PROFILE_TIME, ft_render, render
from lpipsPyTorch import lpips
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

MACRO_ENABLE_SAVE_PROBABILITY_PLOTS_SAVE_HOOK = True

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def get_time_csv_path(csv_path):
    path = Path(csv_path)
    return str(path.with_name(path.stem + "_time" + path.suffix))


def get_time_detail_csv_path(csv_path):
    path = Path(csv_path)
    return str(path.with_name(path.stem + "_time_detail" + path.suffix))


def get_rate_diag_param_groups(gaussians):
    groups = [
        ("opacity", [gaussians._opacity]),
        ("f_dc", [gaussians._features_dc]),
        ("f_rest", [gaussians._features_rest]),
        ("scaling", [gaussians._scaling]),
        ("rotation", [gaussians._rotation]),
    ]
    quantizer_params = gaussians.get_quantizer_trainable_params() if hasattr(gaussians, "get_quantizer_trainable_params") else []
    if quantizer_params:
        groups.append(("quantizers", quantizer_params))
    if hasattr(gaussians, "ans_entropy_bottlenecks") and len(gaussians.ans_entropy_bottlenecks) > 0:
        ans_main_params = [
            param for name, param in gaussians.ans_entropy_bottlenecks.named_parameters()
            if not name.endswith("quantiles")
        ]
        if ans_main_params:
            groups.append(("ans_models", ans_main_params))
    return groups


def run_rate_gradient_diagnostic(
    iteration,
    loss_R,
    current_total_bits,
    num_points,
    viewpoint_cam,
    gaussians,
    pipe,
    background,
    dataset,
):
    if not isinstance(loss_R, torch.Tensor) or not loss_R.requires_grad:
        return

    group_specs = get_rate_diag_param_groups(gaussians)
    flat_params = []
    flat_meta = []
    for group_name, params in group_specs:
        for param in params:
            if param.requires_grad:
                flat_params.append(param)
                flat_meta.append(group_name)

    if not flat_params:
        return

    grads = torch.autograd.grad(
        loss_R,
        flat_params,
        retain_graph=True,
        allow_unused=True,
    )

    group_norm_sq = {}
    originals = []
    for param, grad, group_name in zip(flat_params, grads, flat_meta):
        if grad is None:
            continue
        group_norm_sq[group_name] = group_norm_sq.get(group_name, 0.0) + float((grad.detach() ** 2).sum().item())
        originals.append((param, param.detach().clone(), grad.detach()))

    grad_norms = {
        group_name: value ** 0.5
        for group_name, value in group_norm_sq.items()
    }

    step_size = float(getattr(dataset, "rate_grad_diag_step", 1e-4))
    bits_before = float(current_total_bits.detach().item()) if isinstance(current_total_bits, torch.Tensor) else float(current_total_bits)
    bits_after = bits_before

    try:
        with torch.no_grad():
            for param, _, grad in originals:
                param.add_(grad, alpha=-step_size)

            diag_render_pkg = ft_render(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                training=True,
                raht=dataset.raht,
                debug=False,
                per_channel_quant=dataset.per_channel_quant,
                per_block_quant=dataset.per_block_quant,
                clamp_color=dataset.clamp_color,
            )
            diag_total_bits = diag_render_pkg.get("total_bits", None)
            if isinstance(diag_total_bits, torch.Tensor):
                bits_after = float(diag_total_bits.detach().item())
    finally:
        with torch.no_grad():
            for param, original, _ in originals:
                param.copy_(original)

    bpp_before = bits_before / max(int(num_points), 1)
    bpp_after = bits_after / max(int(num_points), 1)
    grad_summary = ", ".join(
        f"{name}={grad_norms.get(name, 0.0):.3e}"
        for name, _ in group_specs
    )
    print(
        f"[RATE-DIAG][ITER {iteration}] grad_norms: {grad_summary}"
    )
    print(
        f"[RATE-DIAG][ITER {iteration}] virtual_step={step_size:.1e} | "
        f"bits: {bits_before:.2f} -> {bits_after:.2f} "
        f"(delta={bits_after - bits_before:+.2f}) | "
        f"bpp: {bpp_before:.6f} -> {bpp_after:.6f}"
    )

def cal_sens(
        gaussians,
        views,
        pipeline, 
        background,
    ):
    scaling = gaussians.get_scaling.detach()
    cov3d = gaussians.covariance_activation(
        torch.tensor(scaling, device='cuda'), 1.0, torch.tensor(gaussians.get_rotation.detach(), device='cuda')
    ).requires_grad_(True)
    h1 = gaussians._features_dc.register_hook(lambda grad: grad.abs())
    h2 = gaussians._features_rest.register_hook(lambda grad: grad.abs())
    h3 = cov3d.register_hook(lambda grad: grad.abs())
    gaussians._features_dc.grad = None
    gaussians._features_rest.grad = None
    num_pixels = 0

    for view in tqdm(views, desc="Calculating c3dgs importance"):
        rendering = render(
            view,
            gaussians,
            pipeline,
            background,
            cov3d=cov3d,
            debug=False,
            clamp_color=False,
            meson_count=False,
            f_count=False
        )["render"]
        loss = rendering.sum()
        loss.backward()
        num_pixels += rendering.shape[1]*rendering.shape[2]
    importance = torch.cat(
        [gaussians._features_dc.grad, gaussians._features_rest.grad],
        1,
    ).flatten(-2)/num_pixels
    cov_grad = cov3d.grad/num_pixels
    h1.remove()
    h2.remove()
    h3.remove()
    torch.cuda.empty_cache()
    color_imp = importance.detach()
    cov_imp = cov_grad.detach()
    color_imp_n = color_imp.amax(-1)
    cov_imp_n = cov_imp.amax(-1)
    return color_imp_n * torch.pow(cov_imp_n, args.cov_beta)

def pre_volume(volume, beta):
    # volume = torch.tensor(volume)
    index = int(volume.shape[0] * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    v_list = torch.pow(volume / kth_percent_largest, beta)
    return v_list

def cal_imp(
        gaussians,
        views,
        pipeline, 
        background
    ):
    beta_list = {
        'chair': 0.03,
        'drums': 0.05,
        'ficus': 0.03,
        'hotdog': 0.03,
        'lego': 0.05,
        'materials': 0.03,
        'mic': 0.03,
        'ship': 0.03,
        'bicycle': 0.03,
        'bonsai': 0.1,
        'counter': 0.1,
        'garden': 0.1,
        'kitchen': 0.1,
        'room': 0.1,
        'stump': 0.01,
    }   
    
    full_opa_imp = None 

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_results = render(view, gaussians, pipeline, background, debug=True, clamp_color=True, meson_count=True, f_count=False, depth_count=False)
        if full_opa_imp is not None:
            full_opa_imp += render_results["imp"]
        else:
            full_opa_imp = render_results["imp"]
            
        del render_results
        
    volume = torch.prod(gaussians.get_scaling, dim=1)

    v_list = pre_volume(volume, beta_list.get(pipeline.scene_imp, 0.1))
    imp = v_list * full_opa_imp
    
    return imp.detach()
    
def prune_mask(percent, imp):
    sorted_tensor, _ = torch.sort(imp, dim=0)
    index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
    value_nth_percentile = sorted_tensor[index_nth_percentile]
    prune_mask = (imp <= value_nth_percentile).squeeze()
    return prune_mask

lr_scale_list = {
    'chair': 0.4,
    'drums': 0.4,
    'ficus': 0.4,
    'hotdog': 0.1,
    'lego': 0.4,
    'materials': 0.1,
    'mic': 0.1,
    'ship': 0.1,
    'bicycle': 0.15,
    'bonsai': 0.1,
    'counter': 0.1,
    'garden': 0.1,
    'kitchen': 0.1,
    'room': 0.1,
    'stump': 0.01,
    'flowers': 0.05,
    'treehill': 0.05,
    'drjohnson': 0.1,
    'playroom': 0.1,
    'train': 0.05,
    'truck': 0.2
}


universal_config = {
    # 'lseg': {
    #     'chair': 4000,
    #     'drums': 4000,
    #     'ficus': 4000,
    #     'hotdog': 2000,
    #     'lego': 2000,
    #     'materials': 200,
    #     'mic': 4000,
    #     'ship': 2000,
    #     'bicycle': 20000,
    #     'bonsai': 20000,
    #     'counter': 5000,
    #     'garden': 20000,
    #     'kitchen': 1000,
    #     'room': 20000,
    #     'stump': 20000,
    #     'drjohnson': 20000,
    #     'playroom': 20000,
    #     'truck': 25000,
    #     'train': 3000
    # },
    'n_block': {
        'bicycle' : 66,
        'bonsai' : 66,
        'counter' : 66,
        'garden' : 66,
        'kitchen' : 66,
        'room': 66,
        'stump': 66,
        'flowers': 66,
        'treehill': 66,
        'drjohnson': 66,
        'playroom': 66,
        'train': 66,
        'truck': 66,
        'chair': 62,
        'drums': 66,
        'ficus': 50,
        'hotdog': 66,
        'lego': 66,
        'materials': 66,
        'mic': 52,
        'ship': 66
    },
    'cb': {
        'chair': 2048,
        'drums': 2048,
        'ficus': 2048,
        'hotdog': 4096,
        'lego': 4096,
        'materials': 4096,
        'mic': 2048,
        'ship': 8192,
        'bicycle': 2048,
        'bonsai': 2048,
        'counter': 2048,
        'garden': 2048,
        'kitchen': 8192,
        'room': 2048,
        'stump': 2048,
        'flowers': 2048,
        'treehill': 2048,
        'drjohnson': 2048,
        'playroom': 2048,
        'train': 4096,
        'truck': 4096
    },
    'depth': {
        'chair': 14,
        'drums': 14,
        'ficus': 14,
        'hotdog': 14,
        'lego': 14,
        'materials': 14,
        'mic': 14,
        'ship': 14,
        'bicycle': 20,
        'bonsai': 19,
        'counter': 19,
        'garden': 20,
        'kitchen': 19,
        'room': 19,
        'stump': 20,
        'flowers': 20,
        'treehill': 20,
        'drjohnson': 20,
        'playroom': 20,
        'train': 20,
        'truck': 20
    },
    'prune':  {
        'chair': 0.06,
        'drums': 0.18,
        'ficus': 0.28,
        'hotdog':  0.32,
        'lego': 0.1,
        'materials': 0.1,
        'mic': 0.3,
        'ship': 0.18,
        'bicycle': 0.3,
        'bonsai': 0.24,
        'counter': 0.12,
        'garden': 0.18,
        'kitchen': 0.14,
        'room': 0.22,
        'stump': 0.2,
        'flowers': 0.2,
        'treehill': 0.2,
        'drjohnson': 0.41,
        'playroom': 0.0,
        'train': 0.12,
        'truck': 0.38
    }
}

config2 = {
    # 'lseg': {
    #     'chair': 4000,
    #     'drums': 4000,
    #     'ficus': 4000,
    #     'hotdog': 2000,
    #     'lego': 2000,
    #     'materials': 200,
    #     'mic': 4000,
    #     'ship': 2000,
    #     'bicycle': 20000,
    #     'bonsai': 20000,
    #     'counter': 5000,
    #     'garden': 20000,
    #     'kitchen': 1000,
    #     'room': 20000,
    #     'stump': 20000,
    #     'drjohnson': 20000,
    #     'playroom': 20000,
    #     'truck': 25000,
    #     'train': 3000
    # },
    'n_block': {
        'bicycle' : 50,
        'bonsai' : 50,
        'counter' : 50,
        'garden' : 50,
        'kitchen' : 50,
        'room': 50,
        'stump': 50,
        'flowers': 50,
        'treehill': 50,
        'drjohnson': 50,
        'playroom': 50,
        'train': 50,
        'truck': 50,
        'chair': 48,
        'drums': 50,
        'ficus': 50,
        'hotdog': 50,
        'lego': 50,
        'materials': 50,
        'mic': 42,
        'ship': 50
    },
    'cb': {
        'chair': 2048,
        'drums': 2048,
        'ficus': 2048,
        'hotdog': 4096,
        'lego': 4096,
        'materials': 4096,
        'mic': 2048,
        'ship': 8192,
        'bicycle': 2048,
        'bonsai': 2048,
        'counter': 2048,
        'garden': 2048,
        'kitchen': 8192,
        'room': 2048,
        'stump': 2048,
        'flowers': 2048,
        'treehill': 2048,
        'drjohnson': 2048,
        'playroom': 2048,
        'train': 4096,
        'truck': 4096
    },
    'depth': {
        'chair': 14,
        'drums': 14,
        'ficus': 14,
        'hotdog': 14,
        'lego': 14,
        'materials': 14,
        'mic': 14,
        'ship': 14,
        'bicycle': 20,
        'bonsai': 19,
        'counter': 19,
        'garden': 20,
        'kitchen': 19,
        'room': 19,
        'stump': 20,
        'flowers': 20,
        'treehill': 20,
        'drjohnson': 20,
        'playroom': 20,
        'train': 20,
        'truck': 20
    },
    'prune':  {
        'chair': 0.06,
        'drums': 0.18,
        'ficus': 0.28,
        'hotdog':  0.32,
        'lego': 0.1,
        'materials': 0.1,
        'mic': 0.3,
        'ship': 0.18,
        'bicycle': 0.3,
        'bonsai': 0.24,
        'counter': 0.12,
        'garden': 0.18,
        'kitchen': 0.14,
        'room': 0.22,
        'stump': 0.2,
        'flowers': 0.45,
        'treehill': 0.45,
        'drjohnson': 0.41,
        'playroom': 0.0,
        'train': 0.12,
        'truck': 0.38
    }
}


# config3 = {
#     'n_block': {
#         'bicycle' : 57,
#         'bonsai' : 57,
#         'counter' : 57,
#         'garden' : 57,
#         'kitchen' : 57,
#         'room': 57,
#         'stump': 57,
#         'flowers': 57,
#         'treehill': 57,
#         'drjohnson': 57,
#         'playroom': 57,
#         'train': 57,
#         'truck': 57,
#         'chair': 52,
#         'drums': 57,
#         'ficus': 57,
#         'hotdog': 57,
#         'lego': 57,
#         'materials': 57,
#         'mic': 48,
#         'ship': 57
#     },
#     'cb': {
#         'chair': 2048,
#         'drums': 2048,
#         'ficus': 2048,
#         'hotdog': 4096,
#         'lego': 4096,
#         'materials': 4096,
#         'mic': 2048,
#         'ship': 8192,
#         'bicycle': 2048,
#         'bonsai': 2048,
#         'counter': 2048,
#         'garden': 2048,
#         'kitchen': 8192,
#         'room': 2048,
#         'stump': 2048,
#         'flowers': 2048,
#         'treehill': 2048,
#         'drjohnson': 2048,
#         'playroom': 2048,
#         'train': 4096,
#         'truck': 4096
#     },
#     'depth': {
#         'chair': 14,
#         'drums': 14,
#         'ficus': 14,
#         'hotdog': 14,
#         'lego': 14,
#         'materials': 14,
#         'mic': 14,
#         'ship': 14,
#         'bicycle': 20,
#         'bonsai': 19,
#         'counter': 19,
#         'garden': 20,
#         'kitchen': 19,
#         'room': 19,
#         'stump': 20,
#         'flowers': 20,
#         'treehill': 20,
#         'drjohnson': 20,
#         'playroom': 20,
#         'train': 20,
#         'truck': 20
#     },
#     'prune':  {
#         'chair': 0.16,
#         'drums': 0.28,
#         'ficus': 0.38,
#         'hotdog':  0.42,
#         'lego': 0.2,
#         'materials': 0.2,
#         'mic': 0.4,
#         'ship': 0.28,
#         'bicycle': 0.4,
#         'bonsai': 0.34,
#         'counter': 0.22,
#         'garden': 0.28,
#         'kitchen': 0.24,
#         'room': 0.32,
#         'stump': 0.3,
#         'flowers': 0.5,
#         'treehill': 0.5,
#         'drjohnson': 0.41,
#         'playroom': 0.2,
#         'train': 0.22,
#         'truck': 0.4
#     }
# }

config3 = {
    'n_block': {
        'bicycle' : 57,
        'bonsai' : 57,
        'counter' : 57,
        'garden' : 57,
        'kitchen' : 57,
        'room': 57,
        'stump': 57,
        'flowers': 57,
        'treehill': 57,
        'drjohnson': 57,
        'playroom': 57,
        'train': 57,
        'truck': 57,
        'chair': 52,
        'drums': 57,
        'ficus': 57,
        'hotdog': 57,
        'lego': 57,
        'materials': 57,
        'mic': 48,
        'ship': 57
    },
    'cb': {
        'chair': 2048,
        'drums': 2048,
        'ficus': 2048,
        'hotdog': 4096,
        'lego': 4096,
        'materials': 4096,
        'mic': 2048,
        'ship': 8192,
        'bicycle': 2048,
        'bonsai': 2048,
        'counter': 2048,
        'garden': 2048,
        'kitchen': 8192,
        'room': 2048,
        'stump': 2048,
        'flowers': 2048,
        'treehill': 2048,
        'drjohnson': 2048,
        'playroom': 2048,
        'train': 4096,
        'truck': 4096
    },
    'depth': {
        'chair': 14,
        'drums': 14,
        'ficus': 14,
        'hotdog': 14,
        'lego': 14,
        'materials': 14,
        'mic': 14,
        'ship': 14,
        'bicycle': 20,
        'bonsai': 19,
        'counter': 19,
        'garden': 20,
        'kitchen': 19,
        'room': 19,
        'stump': 20,
        'flowers': 20,
        'treehill': 20,
        'drjohnson': 20,
        'playroom': 20,
        'train': 20,
        'truck': 20
    },
    'prune':  {
        'chair': 0.16,
        'drums': 0.28,
        'ficus': 0.38,
        'hotdog':  0.42,
        'lego': 0.2,
        'materials': 0.2,
        'mic': 0.4,
        'ship': 0.28,
        'bicycle': 0.4,
        'bonsai': 0.34,
        'counter': 0.22,
        'garden': 0.28,
        'kitchen': 0.24,
        'room': 0.32,
        'stump': 0.3,
        'flowers': 0.5,
        'treehill': 0.5,
        'drjohnson': 0.41,
        'playroom': 0.2,
        'train': 0.22,
        'truck': 0.4
    }
}

#add by sparseRAHT
config4 = {
    'n_block': {
        'bicycle' : 57,
        'bonsai' : 57,
        'counter' : 57,
        'garden' : 57,
        'kitchen' : 57,
        'room': 57,
        'stump': 57,
        'flowers': 57,
        'treehill': 57,
        'drjohnson': 57,
        'playroom': 57,
        'train': 57,
        'truck': 57,
        'chair': 52,
        'drums': 57,
        'ficus': 57,
        'hotdog': 57,
        'lego': 57,
        'materials': 57,
        'mic': 48,
        'ship': 57
    },
    'cb': {
        'chair': 2048,
        'drums': 2048,
        'ficus': 2048,
        'hotdog': 4096,
        'lego': 4096,
        'materials': 4096,
        'mic': 2048,
        'ship': 8192,
        'bicycle': 2048,
        'bonsai': 2048,
        'counter': 2048,
        'garden': 2048,
        'kitchen': 8192,
        'room': 2048,
        'stump': 2048,
        'flowers': 2048,
        'treehill': 2048,
        'drjohnson': 2048,
        'playroom': 2048,
        'train': 4096,
        'truck': 4096
    },
    'depth': {
        'chair': 14,
        'drums': 14,
        'ficus': 14,
        'hotdog': 14,
        'lego': 14,
        'materials': 14,
        'mic': 14,
        'ship': 14,
        'bicycle': 20,
        'bonsai': 19,
        'counter': 19,
        'garden': 20,
        'kitchen': 19,
        'room': 19,
        'stump': 20,
        'flowers': 20,
        'treehill': 20,
        'drjohnson': 20,
        'playroom': 20,
        'train': 20,
        'truck': 20
    },
    'prune':  {
        'chair': 0.16,
        'drums': 0.28,
        'ficus': 0.38,
        'hotdog':  0.42,
        'lego': 0.2,
        'materials': 0.2,
        'mic': 0.4,
        'ship': 0.28,
        'bicycle': 0.4,
        'bonsai': 0.34,
        'counter': 0.22,
        'garden': 0.28,
        'kitchen': 0.24,
        'room': 0.32,
        'stump': 0.3,
        'flowers': 0.5,
        'treehill': 0.5,
        'drjohnson': 0.41,
        'playroom': 0.2,
        'train': 0.22,
        'truck': 0.4
    }
}


# 'prune':  {
#         'chair': 0.06,
#         'drums': 0.18,
#         'ficus': 0.28,
#         'hotdog':  0.32,
#         'lego': 0.1,
#         'materials': 0.1,
#         'mic': 0.3,
#         'ship': 0.18,
#         'bicycle': 0.3,
#         'bonsai': 0.24,
#         'counter': 0.12,
#         'garden': 0.18,
#         'kitchen': 0.14,
#         'room': 0.22,
#         'stump': 0.2,
#         'drjohnson': 0.41,
#         'playroom': 0.0,
#         'train': 0.12,
#         'truck': 0.38
#     }


cb_const = 256
prune_const = 0.66
nerf_syn_small_config = {
    'lseg': {
        'chair': 1000000,
        'drums': 1000000,
        'ficus': 1000000,
        'hotdog': 1000000,
        'lego': 1000000,
        'materials': 1000000,
        'mic': 1000000,
        'ship': 1000000
    },
    'cb': {
        'chair': cb_const,
        'drums': cb_const,
        'ficus': cb_const,
        'hotdog': cb_const,
        'lego': cb_const,
        'materials': cb_const,
        'mic': cb_const,
        'ship': cb_const
    },
    'depth': {
        'chair': 12,
        'drums': 12,
        'ficus': 12,
        'hotdog': 12,
        'lego': 12,
        'materials': 12,
        'mic': 12,
        'ship': 12
    },
    'prune': {
        'chair': prune_const,
        'drums': prune_const,
        'ficus': prune_const,
        'hotdog': prune_const,
        'lego': prune_const,
        'materials': prune_const,
        'mic': prune_const,
        'ship': prune_const
    }
}

def evaluate_test(scene, dataset, pipe, background, iteration=0):
    model_path = dataset.model_path
    cams = scene.getTestCameras()

    ssims = []
    lpipss = []
    psnrs = []

    render_path = os.path.join(model_path, 'test', f'iter_{iteration}')
    os.makedirs(render_path, exist_ok=True)
    
    gts_path = os.path.join(model_path, 'gt', f'iter_{iteration}')
    os.makedirs(gts_path, exist_ok=True)

    for idx, viewpoint in enumerate(tqdm(cams)):
        image = ft_render(
            viewpoint, 
            scene.gaussians, 
            pipe, 
            background,
            training=False,
            raht=dataset.raht,
            debug=dataset.debug,
            per_channel_quant=dataset.per_channel_quant,
            per_block_quant=dataset.per_block_quant,
            clamp_color=True)["render"]
        gt_image = viewpoint.original_image[0:3, :, :].to("cuda")
        # print(gt_image.max(), gt_image.min())
        torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        psnrs.append(psnr(image.unsqueeze(0), gt_image).unsqueeze(0))
        
        ssims.append(ssim(image, gt_image))
        lpipss.append(lpips(image, gt_image, net_type='vgg'))

    
    psnr_val = torch.tensor(psnrs).mean()
    ssim_val = torch.tensor(ssims).mean()
    lpips_val = torch.tensor(lpipss).mean()
    
    return psnr_val.item(), ssim_val.item(), lpips_val.item()


def training(dataset, opt, pipe, testing_iterations, given_ply_path=None):
    # print('dataset.eval', dataset.eval)
    print('debug', dataset.debug)

    # magnify the lr scale of the covergence process is too slow.
    opt.finetune_lr_scale = lr_scale_list[pipe.scene_imp] * 4 
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, depth=dataset.depth, num_bits=dataset.num_bits)
    scene = Scene(dataset, gaussians, given_ply_path=given_ply_path)
    
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # nspd_mask = gaussians.check_spd()
    # print('nspd_mask.shape', nspd_mask.shape)
    
    print("\n" + "-"*50)
    print("【Step1】重要性剪枝")
    print("-"*50)
    original_points = gaussians.get_xyz.shape[0]
    
    with torch.no_grad():
        imp = cal_imp(gaussians, scene.getTrainCameras(), pipe, background)
    # imp = cal_sens(gaussians, scene.getTrainCameras(), pipe, background)
    
    print(f"重要性分数范围: [{imp.min().item():.6f}, {imp.max().item():.6f}]")
    print(f"重要性分数均值: {imp.mean().item():.6f}")
    print(f"剪枝阈值: {dataset.percent*100:.1f}%")
    
    pmask = prune_mask(dataset.percent, imp)  #通过修改掩码剪枝
    imp = imp[torch.logical_not(pmask)]
    gaussians.prune_points(pmask)
    
    remaining_points = gaussians.get_xyz.shape[0]
    print(f"剪枝: {original_points:,} → {remaining_points:,} 点 ({dataset.percent*100:.0f}% pruned)")

    
    print("\n" + "-"*50)
    print("【Step2】八叉树编码 & 量化器初始化")
    print("-"*50)
    print(f"  八叉树: depth={dataset.depth}, RAHT={'ON' if dataset.raht else 'OFF'}")
    
    gaussians.octree_coding(
        imp,
        dataset.oct_merge,
        raht=dataset.raht
    )
    
    if dataset.per_channel_quant:
        print(f"  量化模式：per_channel_quant")
        dataset.n_block = 1    #一个属性只用一个量化器，不使用之前定义的n_block
        # 配置不同属性的量化位数
        bit_config = {
            'opacity': 8,
            'euler': 8,
            'scale': 10,
            'f_dc': 8,
            'f_rest_0': 4,
            'f_rest_1': 4,
            'f_rest_2': 2,
        }
        gaussians.init_qas(
            dataset.n_block,
            bit_config=bit_config,
            quant_type=dataset.quant_type,
            encode=getattr(dataset, 'encode', 'deflate'),
            ans_subgroup_count=getattr(dataset, 'ans_subgroup_count', 1),
        )
        print(f"  通道量化: 启用")
    elif dataset.per_block_quant:
        print(f"  量化模式：per_block_quant")
        bit_config = {
            'opacity': 8,       # alpha
            'euler': 8,         # rotation (欧拉角)
            'scale': 10,        # 需要更高精度
            'f_dc': 8,          # sh_0
            'f_rest_0': 4,      # sh_1 (SH degree 1: 9维)
            'f_rest_1': 4,      # sh_2 (SH degree 2: 15维)
            'f_rest_2': 2,      # sh_3 (SH degree 3: 21维)
        }
        gaussians.init_qas(
            dataset.n_block,
            bit_config=bit_config,
            quant_type=dataset.quant_type,
            encode=getattr(dataset, 'encode', 'deflate'),
            ans_subgroup_count=getattr(dataset, 'ans_subgroup_count', 1),
        )
    else:
        print(f"未知的量化模式")

    # VQ训练（本项目不使用VQ，已删去VQ内部函数，进入vq_fe函数后会直接返回）
    # gaussians.vq_fe(imp, dataset.codebook_size, dataset.batch_size, dataset.steps)
    
        
    print("\n" + "-"*50)
    print("【Step3】初始评估 (Iter 0)")
    print("-"*50)
    with torch.no_grad():
        psnr_val, ssim_val, lpips_val = evaluate_test(
            scene,
            dataset,
            pipe,
            background,
            iteration=0
        )
        
        #保存压缩文件
        if getattr(dataset, 'encode', 'deflate').lower() == "ans" and hasattr(gaussians, 'ans_entropy_bottlenecks'):
            for eb in gaussians.ans_entropy_bottlenecks.values():
                eb.update(force=True)
        zip_size = scene.save_ft(
            "0",
            pipe,
            per_channel_quant=dataset.per_channel_quant,
            per_block_quant=dataset.per_block_quant,
            bit_packing=dataset.bit_packing,
            export_ans_offline_fit=getattr(dataset, 'export_ans_offline_fit', False),
            export_ans_offline_fit_steps=getattr(dataset, 'export_ans_offline_fit_steps', 1000),
            export_ans_offline_fit_main_lr=getattr(dataset, 'export_ans_offline_fit_main_lr', 1e-3),
            export_ans_offline_fit_aux_lr=getattr(dataset, 'export_ans_offline_fit_aux_lr', 1e-3),
            save_probability_plots=(
                getattr(dataset, 'save_probability_plots', False)
                if MACRO_ENABLE_SAVE_PROBABILITY_PLOTS_SAVE_HOOK else False
            ),
        )
        zip_size = zip_size / 1024 / 1024 # to MB
        
        print("\n" + "-"*70)
        print("【初始评估结果】")
        print("-"*70)
        print(f"PSNR:  {psnr_val:.4f} dB")
        print(f"SSIM:  {ssim_val:.4f}")
        print(f"LPIPS: {lpips_val:.4f}")
        print(f"文件大小: {zip_size:.2f} MB")
        print("-"*70)
        
        row = []
        row.append(pipe.scene_imp)
        row.extend([0, psnr_val, ssim_val, lpips_val, zip_size])
        f = open(dataset.csv_path, 'a+')
        wtr = csv.writer(f)
        wtr.writerow(row)
        f.close()
    
    print("\n" + "-"*50)
    print("【Step4】微调训练")
    print("-"*50)
    print(f"总迭代次数: {opt.iterations}")
    print(f"学习率缩放: {opt.finetune_lr_scale}")
    print(f"测试迭代: {testing_iterations}")
    print(f"稀疏性损失权重 (λ_s): {dataset.lambda_sparsity}")
    print(f"码率损失权重 (λ_r): {dataset.lambda_rate}")
    encode_mode = getattr(dataset, 'encode', 'deflate').lower()
    uses_rate_model = encode_mode in ("ans", "laplace")
    if dataset.lambda_sparsity > 0:
        print(f"  启用稀疏性正则化，促进RAHT系数稀疏化")
    if encode_mode == "ans" and dataset.lambda_rate > 0:
        print(f"  启用ANS码率约束，直接优化量化块的熵估计")
    if encode_mode == "laplace" and dataset.lambda_rate > 0:
        print(f"  启用拉普拉斯码率约束，按块估计零均值分布尺度")
    if getattr(dataset, 'rate_grad_diag', False):
        print(
            f"  启用loss_R梯度诊断: interval={int(getattr(dataset, 'rate_grad_diag_interval', 200))}, "
            f"virtual_step={float(getattr(dataset, 'rate_grad_diag_step', 1e-4)):.1e}"
        )
    if uses_rate_model:
        if dataset.lambda_rate > 0 and dataset.lambda_sparsity > 0:
            print("  当前损失模式: hybrid (loss_D + λ_r * loss_R + λ_s * loss_S)")
        elif dataset.lambda_rate > 0:
            print("  当前损失模式: rate-only (loss_D + λ_r * loss_R)")
        elif dataset.lambda_sparsity > 0:
            print("  当前损失模式: sparsity-only (loss_D + λ_s * loss_S)")
        else:
            print("  当前损失模式: distortion-only (loss_D)")
    print("="*70 + "\n")
    
    gaussians.finetuning_setup(opt) #微调，需要固定位置，只对外观微调

    # ==========================================
    # 初始化 ANS 的辅助优化器 (拟合 CDF 曲线)
    # ==========================================
    aux_optimizer = None
    if encode_mode == "ans" and hasattr(gaussians, 'ans_entropy_bottlenecks'):
        aux_quantiles = [eb.quantiles for eb in gaussians.ans_entropy_bottlenecks.values()]
        if aux_quantiles:
            aux_optimizer = torch.optim.Adam(aux_quantiles, lr=1e-3)
            print(f"\n【ANS 编码】已启用，初始化 aux_optimizer 管理 {len(aux_quantiles)} 个 CDF 参数")

    # 输出初始LSQ scale参数
    if dataset.per_block_quant and dataset.quant_type.lower() == "lsq":
        gaussians.print_lsq_scale_evolution(0)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
        
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(opt.iterations), desc="微调进度")
    psnr_train = 0
    for iteration in range(1, opt.iterations + 1):    
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        # try:
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        
        render_pkg = ft_render(
            viewpoint_cam, 
            gaussians, 
            pipe, 
            background,
            training=True,
            raht=dataset.raht,
            debug=dataset.debug,
            per_channel_quant=dataset.per_channel_quant,
            per_block_quant=dataset.per_block_quant,
            clamp_color=dataset.clamp_color)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        current_profile = render_pkg.get("profile") if PROFILE_TIME else None

        # Loss
        if PROFILE_TIME:
            torch.cuda.synchronize()
            t_loss_start = torch.cuda.Event(enable_timing=True)
            t_loss_end = torch.cuda.Event(enable_timing=True)
            t_loss_start.record()
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        
        # 数据保真损失 (Data fidelity loss)
        loss_D = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # 稀疏性损失 (Sparsity loss)
        loss_S = 0.0
        if dataset.lambda_sparsity > 0 and "raht_coeffs" in render_pkg:
            # 获取RAHT AC系数
            raht_ac = render_pkg["raht_coeffs"]  # shape: [n_ac, n_ch]
            
            # 计算稀疏性损失: L_S = (1/n_ch) * Σ_i Σ_j |x_ij|
            # 其中:
            #   n_ch: 通道数（高斯属性数量）= 55
            #   n_ac: AC系数数量 = 点数 - 1
            #   x_ij: 第i个通道的第j个AC系数
            # 该公式对应于每个通道的平均L1范数
            n_ch = raht_ac.shape[1]  # 通道数 (55)
            n_ac = raht_ac.shape[0]   # AC系数数量
            
            # 实现: (1/n_ch) * sum(|x_ij|) for all i,j
            # torch.abs(raht_ac).sum() 计算所有|x_ij|的总和
            # 然后除以n_ch得到每个通道的平均L1范数
            loss_S = torch.abs(raht_ac).sum() / n_ch
        # print(f"迭代 {iteration}: AC_loss = {loss_S.item():.2e} (初始值)")
        # ==========================================
        # 提取概率并计算 Rate Loss (码率约束)
        # ==========================================
        loss_R = torch.tensor(0.0, device="cuda")

        if "total_bits" in render_pkg:
            total_bits = render_pkg["total_bits"]
            num_points = gaussians.get_xyz.shape[0]
            if num_points > 0:
                if iteration % 10 == 0:
                    print(f"迭代 {iteration}: 当前总比特数 = {total_bits.item():.2f} bits, 平均每点 = {total_bits.item()/num_points:.4f} bits/point")
                loss_R = total_bits / num_points
        
        # ==========================================
        # 组装最终 Loss
        # λ_s 和 λ_r 直接控制两类约束强度：
        #   λ_s > 0, λ_r = 0 -> sparsity-only
        #   λ_s = 0, λ_r > 0 -> rate-only
        #   λ_s > 0, λ_r > 0 -> hybrid
        # ==========================================
        loss = loss_D
        if dataset.lambda_sparsity > 0:
            loss = loss + dataset.lambda_sparsity * loss_S
        if uses_rate_model and dataset.lambda_rate > 0:
            loss = loss + dataset.lambda_rate * loss_R
        if PROFILE_TIME:
            t_loss_end.record()

        should_run_rate_diag = (
            getattr(dataset, 'rate_grad_diag', False)
            and uses_rate_model
            and dataset.lambda_rate > 0
            and iteration % max(int(getattr(dataset, 'rate_grad_diag_interval', 200)), 1) == 0
            and "total_bits" in render_pkg
        )
        if should_run_rate_diag:
            run_rate_gradient_diagnostic(
                iteration,
                loss_R,
                render_pkg["total_bits"],
                gaussians.get_xyz.shape[0],
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                dataset,
            )
            
        # 清空梯度
        # if aux_optimizer is not None:
        #     aux_optimizer.zero_grad(set_to_none=True)
        if PROFILE_TIME:
            torch.cuda.synchronize()
            t_main_backward_start = torch.cuda.Event(enable_timing=True)
            t_main_backward_end = torch.cuda.Event(enable_timing=True)
            t_main_backward_start.record()
        loss.backward()
        if PROFILE_TIME:
            t_main_backward_end.record()

        # ==========================================
        # ANS 辅助参数的反向传播
        # ==========================================
        aux_backward_ms = 0.0
        if aux_optimizer is not None:
            if PROFILE_TIME:
                torch.cuda.synchronize()
                t_aux_backward_start = torch.cuda.Event(enable_timing=True)
                t_aux_backward_end = torch.cuda.Event(enable_timing=True)
                t_aux_backward_start.record()
            aux_optimizer.zero_grad(set_to_none=True)
            aux_loss = torch.tensor(0.0, device="cuda")
            for eb in gaussians.ans_entropy_bottlenecks.values():
                aux_loss = aux_loss + eb.loss()
            if isinstance(aux_loss, torch.Tensor) and aux_loss.requires_grad:
                aux_loss.backward()
                aux_optimizer.step()
            if PROFILE_TIME:
                t_aux_backward_end.record()

        if PROFILE_TIME:
            iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix_dict = {
                    "损失": f"{ema_loss_for_log:.6f}",
                    "L1": f"{Ll1.item():.6f}"
                }
                # 如果使用稀疏性损失，显示它
                if dataset.lambda_sparsity > 0 and isinstance(loss_S, torch.Tensor):
                    postfix_dict["稀疏"] = f"{loss_S.item():.2e}"
                if uses_rate_model and isinstance(loss_R, torch.Tensor):
                    postfix_dict["码率loss"] = f"{loss_R.item():.2e}"
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(10)
            
            # # 每100次迭代输出LSQ scale参数演化
            # if iteration % 100 == 0:
            #     gaussians.print_lsq_scale_evolution(iteration)
            
            # # 每50次迭代输出简要统计信息
            # if iteration % 50 == 0 and dataset.per_block_quant and dataset.quant_type.lower() == "lsq":
            #     gaussians.print_lsq_scale_summary(iteration)
            # if iteration == opt.iterations:
            #     progress_bar.close()
            #     if dataset.lambda_sparsity > 0:
            #         print(f"\n微调完成！最终损失: {ema_loss_for_log:.6f} (包含稀疏性正则化)")
            #     else:
            #         print(f"\n微调完成！最终损失: {ema_loss_for_log:.6f}")

            # Keep track of max radii in image-space for pruning
            # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            elapsed_ms = iter_start.elapsed_time(iter_end) if PROFILE_TIME else None
            cur_psnr, _, _ = training_report(
                tb_writer, 
                iteration, 
                Ll1, 
                loss, 
                l1_loss, 
                elapsed_ms,
                testing_iterations, 
                scene, 
                ft_render, 
                pipe.scene_imp, 
                dataset.csv_path, 
                get_time_csv_path(dataset.csv_path),
                get_time_detail_csv_path(dataset.csv_path),
                dataset.model_path, 
                current_profile,
                (pipe, background, False, 1.0, None, 
                 dataset.raht, dataset.per_channel_quant, 
                 dataset.per_block_quant, False, True))
            
            if cur_psnr > psnr_train:
                psnr_train = cur_psnr
                print("\n Saving best Gaussians on Train Set.")
                if getattr(dataset, 'encode', 'deflate').lower() == "ans" and hasattr(gaussians, 'ans_entropy_bottlenecks'):
                    for eb in gaussians.ans_entropy_bottlenecks.values():
                        eb.update(force=True)
                scene.save_ft(
                    'best',
                    pipe,
                    per_channel_quant=dataset.per_channel_quant,
                    per_block_quant=dataset.per_block_quant,
                    bit_packing=dataset.bit_packing,
                    export_ans_offline_fit=getattr(dataset, 'export_ans_offline_fit', False),
                    export_ans_offline_fit_steps=getattr(dataset, 'export_ans_offline_fit_steps', 1000),
                    export_ans_offline_fit_main_lr=getattr(dataset, 'export_ans_offline_fit_main_lr', 1e-3),
                    export_ans_offline_fit_aux_lr=getattr(dataset, 'export_ans_offline_fit_aux_lr', 1e-3),
                    save_probability_plots=(
                        getattr(dataset, 'save_probability_plots', False)
                        if MACRO_ENABLE_SAVE_PROBABILITY_PLOTS_SAVE_HOOK else False
                    ),
                )
 
            
            # Optimizer step
            optimizer_step_ms = 0.0
            if iteration < opt.iterations:
                if PROFILE_TIME:
                    torch.cuda.synchronize()
                    t_opt_start = torch.cuda.Event(enable_timing=True)
                    t_opt_end = torch.cuda.Event(enable_timing=True)
                    t_opt_start.record()
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.update_learning_rate(iteration+30000)
                if PROFILE_TIME:
                    t_opt_end.record()
                    torch.cuda.synchronize()
                    optimizer_step_ms = t_opt_start.elapsed_time(t_opt_end)

            if PROFILE_TIME:
                torch.cuda.synchronize()
                detailed_profile = dict(current_profile or {})
                detailed_profile["loss_compute"] = t_loss_start.elapsed_time(t_loss_end)
                detailed_profile["main_backward"] = t_main_backward_start.elapsed_time(t_main_backward_end)
                if aux_optimizer is not None:
                    aux_backward_ms = t_aux_backward_start.elapsed_time(t_aux_backward_end)
                detailed_profile["aux_backward"] = aux_backward_ms
                detailed_profile["optimizer_step"] = optimizer_step_ms
                current_profile = detailed_profile
                if iteration in testing_iterations:
                    step_total_ms = (elapsed_ms or 0.0) + optimizer_step_ms
                    detail_profile_ms = current_profile or {}
                    detail_row = [
                        pipe.scene_imp,
                        iteration,
                        step_total_ms,
                        float(detail_profile_ms.get("feature_prep", 0.0)) * 1000.0,
                        float(detail_profile_ms.get("raht_forward", 0.0)) * 1000.0,
                        float(detail_profile_ms.get("quant", 0.0)) * 1000.0,
                        float(detail_profile_ms.get("entropy_collect", 0.0)) * 1000.0,
                        float(detail_profile_ms.get("entropy_model", 0.0)) * 1000.0,
                        float(detail_profile_ms.get("entropy", 0.0)) * 1000.0,
                        float(detail_profile_ms.get("raht_inverse", 0.0)) * 1000.0,
                        float(detail_profile_ms.get("covariance", 0.0)) * 1000.0,
                        float(detail_profile_ms.get("sh_eval", 0.0)) * 1000.0,
                        float(detail_profile_ms.get("rasterize_forward", 0.0)) * 1000.0,
                        float(detail_profile_ms.get("loss_compute", 0.0)),
                        float(detail_profile_ms.get("main_backward", 0.0)),
                        float(detail_profile_ms.get("aux_backward", 0.0)),
                        float(detail_profile_ms.get("optimizer_step", 0.0)),
                    ]
                    f = open(get_time_detail_csv_path(dataset.csv_path), 'a+')
                    wtr = csv.writer(f)
                    wtr.writerow(detail_row)
                    f.close()
        # except Exception as e:
        #     print('error but go')

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
        tb_writer, 
        iteration, 
        Ll1, 
        loss, 
        l1_loss, 
        elapsed, 
        testing_iterations,
        scene : Scene, 
        renderFunc, 
        scene_name, 
        csv_path, 
        time_csv_path,
        time_detail_csv_path,
        model_path, 
        profile,
        renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        if PROFILE_TIME and elapsed is not None:
            tb_writer.add_scalar('iter_time', elapsed, iteration)
    psnr_val, ssim_val, lpips_val = 0, 0, 0
    # Report test and samples of training set
    if iteration in testing_iterations:
        # print('trianing report, iteration', iteration)
        torch.cuda.empty_cache()
        config = {
            'train' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            'test' : scene.getTestCameras()
        }
        
        for mode in ['train', 'test']:
            cams = config[mode]
            images = torch.tensor([], device="cuda")
            gts = torch.tensor([], device="cuda")
            
            if mode == 'test':
                ssims = []
                lpipss = []
                
            render_path = os.path.join(model_path, mode, f'iter_{iteration}')
            os.makedirs(render_path, exist_ok=True)
            
            for idx, viewpoint in enumerate(cams):
                if mode == 'test':
                    image = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]
                    torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                    gt_image = viewpoint.original_image[0:3, :, :].to("cuda")
                    torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips(image, gt_image, net_type='vgg'))
                if mode == 'train':
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                images = torch.cat((images, image.unsqueeze(0)), dim=0)
                gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)
            
            l1_test = l1_loss(images, gts)
            psnr_val = psnr(images, gts).mean()
            
            if mode == 'train':           
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, mode, l1_test, psnr_val))
                
                # 打印量化参数
                #if hasattr(scene.gaussians, 'print_quantization_params'):
                #    scene.gaussians.print_quantization_params(iteration=iteration)
                
                if tb_writer:
                    tb_writer.add_scalar(mode + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(mode + '/loss_viewpoint - psnr', psnr_val, iteration)
                torch.cuda.empty_cache()
            
            if mode == 'test':
                ssim_val = torch.tensor(ssims).mean()
                lpips_val = torch.tensor(lpipss).mean()
                
                torch.cuda.empty_cache()
            
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
        zip_size = scene.save_ft(
            iteration,
            pipe,
            per_channel_quant=dataset.per_channel_quant,
            per_block_quant=dataset.per_block_quant,
            bit_packing=dataset.bit_packing,
            export_ans_offline_fit=getattr(dataset, 'export_ans_offline_fit', False),
            export_ans_offline_fit_steps=getattr(dataset, 'export_ans_offline_fit_steps', 1000),
            export_ans_offline_fit_main_lr=getattr(dataset, 'export_ans_offline_fit_main_lr', 1e-3),
            export_ans_offline_fit_aux_lr=getattr(dataset, 'export_ans_offline_fit_aux_lr', 1e-3),
            save_probability_plots=(
                getattr(dataset, 'save_probability_plots', False)
                if MACRO_ENABLE_SAVE_PROBABILITY_PLOTS_SAVE_HOOK else False
            ),
        )
        zip_size = zip_size / 1024 / 1024
        
        row = []
        row.append(scene_name)
        row.extend([iteration, psnr_val.item(), ssim_val.item(), lpips_val.item(), zip_size])
        f = open(csv_path, 'a+')
        wtr = csv.writer(f)
        wtr.writerow(row)
        f.close()

        if PROFILE_TIME and elapsed is not None:
            profile_ms = profile or {}
            time_row = [
                scene_name,
                iteration,
                elapsed,
                float(profile_ms.get("raht_forward", 0.0)) * 1000.0,
                float(profile_ms.get("quant", 0.0)) * 1000.0,
                float(profile_ms.get("entropy", 0.0)) * 1000.0,
                float(profile_ms.get("raht_inverse", 0.0)) * 1000.0,
            ]
            f = open(time_csv_path, 'a+')
            wtr = csv.writer(f)
            wtr.writerow(time_row)
            f.close()
        print("Testset Evaluating {}. PSNR: {}, SSIM: {}, LIPIS: {}".format(iteration, psnr_val, ssim_val, lpips_val, zip_size))

    return psnr_val, ssim_val, lpips_val

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--given_ply_path", default='', type=str)
    args = parser.parse_args(sys.argv[1:])
    
    # Only auto-generate test_iterations if not provided in command line
    # Check if test_iterations was explicitly provided (not default)
    if '--test_iterations' not in sys.argv:
        args.test_iterations = [int(x) for x in range(0, args.iterations+1, 400)]
    
    # Ensure iteration 0 and final iteration are included
    if 0 not in args.test_iterations:
        args.test_iterations = [0] + args.test_iterations
    if args.iterations not in args.test_iterations:
        args.test_iterations.append(args.iterations)
    args.test_iterations = sorted(list(set(args.test_iterations)))
    
    print('args.test_iterations', args.test_iterations)
    
    # args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    
    # use given config
    if pipe.hyper_config == 'universal':
        used_config = universal_config
    elif pipe.hyper_config == 'syn_small':
        used_config = nerf_syn_small_config
    elif pipe.hyper_config == 'config2':
        used_config = config2
    elif pipe.hyper_config == 'config3':
        used_config = config3
    elif pipe.hyper_config == 'config4':
        used_config = config4
    else:
        used_config = None
    # use given config
    
    print('hyper_config', pipe.hyper_config)
    if used_config != None:
        dataset.percent = used_config['prune'][pipe.scene_imp]
        dataset.codebook_size = used_config['cb'][pipe.scene_imp]
        # dataset.lseg = used_config['lseg'][pipe.scene_imp]
        dataset.depth = used_config['depth'][pipe.scene_imp]
        dataset.n_block = used_config['n_block'][pipe.scene_imp]
        
    if not os.path.exists(dataset.csv_path):
        # Create directory if it doesn't exist
        csv_dir = os.path.dirname(dataset.csv_path)
        os.makedirs(csv_dir, exist_ok=True)
        
        f = open(dataset.csv_path, 'a+')
        wtr = csv.writer(f)
        wtr.writerow(['name', 'iteration', 'psnr', 'ssim', 'lpips', 'size'])
        f.close()

    if PROFILE_TIME:
        time_csv_path = get_time_csv_path(dataset.csv_path)
        if not os.path.exists(time_csv_path):
            csv_dir = os.path.dirname(time_csv_path)
            os.makedirs(csv_dir, exist_ok=True)

            f = open(time_csv_path, 'a+')
            wtr = csv.writer(f)
            wtr.writerow([
                'name',
                'iteration',
                'iter_time_ms',
                'raht_forward_ms',
                'quant_ms',
                'entropy_ms',
                'raht_inverse_ms',
            ])
            f.close()

        time_detail_csv_path = get_time_detail_csv_path(dataset.csv_path)
        if not os.path.exists(time_detail_csv_path):
            csv_dir = os.path.dirname(time_detail_csv_path)
            os.makedirs(csv_dir, exist_ok=True)

            f = open(time_detail_csv_path, 'a+')
            wtr = csv.writer(f)
            wtr.writerow([
                'name',
                'iteration',
                'train_step_total_ms',
                'feature_prep_ms',
                'raht_forward_ms',
                'quant_ms',
                'entropy_collect_ms',
                'entropy_model_ms',
                'entropy_total_ms',
                'raht_inverse_ms',
                'covariance_ms',
                'sh_eval_ms',
                'rasterize_forward_ms',
                'loss_compute_ms',
                'main_backward_ms',
                'aux_backward_ms',
                'optimizer_step_ms',
            ])
            f.close()
        
    training(dataset, op.extract(args), pipe, args.test_iterations, args.given_ply_path)

    # All done
    print("\nTraining complete.")
