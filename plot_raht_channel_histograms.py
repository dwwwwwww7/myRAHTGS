import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_entropy_modes import (
    DEFAULT_BIT_CONFIG,
    build_bit_config,
    build_model,
    get_matplotlib_pyplot,
    get_quantization_flags,
    infer_ply_format,
    load_importance,
    quantize_raht_ac,
    sanitize_name,
    warmup_quantizers_from_raht,
)
from scene.gaussian_model import ATTR_GROUP_ORDER, ToEulerAngles_FT, transform_batched_torch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-channel histograms for attributes before RAHT, after RAHT by level, "
            "and after quantization by level."
        )
    )
    parser.add_argument("ply_path", type=Path, help="Input Gaussian PLY file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used to store histogram folders and the summary JSON.",
    )
    parser.add_argument(
        "--ply-format",
        choices=["auto", "quat", "euler"],
        default="auto",
        help="Input PLY rotation format.",
    )
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-bits", type=int, default=8)
    parser.add_argument("--n-block", type=int, default=66)
    parser.add_argument("--quant-type", choices=["lsq", "vanilla"], default="vanilla")
    parser.add_argument("--oct-merge", choices=["mean", "imp", "rand"], default="mean")
    parser.add_argument(
        "--importance",
        type=Path,
        default=None,
        help="Optional .npy/.npz importance file. Required when --oct-merge=imp.",
    )
    parser.add_argument(
        "--quant-granularity",
        choices=["per_block", "per_channel"],
        default="per_block",
        help="Reuse the same quantization layout as training/export.",
    )
    parser.add_argument("--opacity-bits", type=int, default=DEFAULT_BIT_CONFIG["opacity"])
    parser.add_argument("--euler-bits", type=int, default=DEFAULT_BIT_CONFIG["euler"])
    parser.add_argument("--scale-bits", type=int, default=DEFAULT_BIT_CONFIG["scale"])
    parser.add_argument("--f-dc-bits", type=int, default=DEFAULT_BIT_CONFIG["f_dc"])
    parser.add_argument("--f-rest-0-bits", type=int, default=DEFAULT_BIT_CONFIG["f_rest_0"])
    parser.add_argument("--f-rest-1-bits", type=int, default=DEFAULT_BIT_CONFIG["f_rest_1"])
    parser.add_argument("--f-rest-2-bits", type=int, default=DEFAULT_BIT_CONFIG["f_rest_2"])
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=120,
        help="Bin count used for continuous-value histograms.",
    )
    parser.add_argument(
        "--levels-per-row",
        type=int,
        default=4,
        help="How many RAHT levels to place in each row for level-wise figures.",
    )
    parser.add_argument(
        "--summary-name",
        default="raht_histogram_summary.json",
        help="Filename for the JSON summary under --output-dir.",
    )
    return parser.parse_args()


def build_raht_input_matrix(gaussians):
    rotation = gaussians.get_ori_rotation
    norm = torch.sqrt(torch.sum(rotation * rotation, dim=1))
    q = rotation / norm[:, None]
    eulers = ToEulerAngles_FT(q)

    rf = torch.concat(
        [
            gaussians.get_origin_opacity.detach(),
            eulers.detach(),
            gaussians.get_features_dc.detach().contiguous().squeeze(),
            gaussians.get_indexed_feature_extra.detach().contiguous().flatten(-2),
            gaussians.get_ori_scaling.detach(),
        ],
        axis=-1,
    )
    return rf[gaussians.reorder].clone()


def apply_raht_transform(gaussians, values):
    coeffs = values.clone()
    iW1 = gaussians.res["iW1"]
    iW2 = gaussians.res["iW2"]
    iLeft_idx = gaussians.res["iLeft_idx"]
    iRight_idx = gaussians.res["iRight_idx"]

    for depth_idx in range(gaussians.depth * 3):
        coeffs[iLeft_idx[depth_idx]], coeffs[iRight_idx[depth_idx]] = transform_batched_torch(
            iW1[depth_idx],
            iW2[depth_idx],
            coeffs[iLeft_idx[depth_idx]],
            coeffs[iRight_idx[depth_idx]],
        )
    return coeffs


def get_channel_records(gaussians):
    channel_records = []
    for attr_group in ATTR_GROUP_ORDER:
        dims = gaussians.get_attr_group_dims(attr_group)
        for local_idx, dim_idx in enumerate(dims):
            channel_records.append(
                {
                    "attr_group": attr_group,
                    "global_dim": int(dim_idx),
                    "local_channel_index": int(local_idx),
                    "channel_name": f"{attr_group}_c{local_idx:02d}",
                }
            )
    return channel_records


def select_tick_positions(x_values, max_ticks=17):
    if x_values.size <= max_ticks:
        return x_values
    indices = np.linspace(0, x_values.size - 1, num=max_ticks, dtype=int)
    return x_values[indices]


def plot_histogram(values, title, save_path, bins):
    plt = get_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    if values.size == 0:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        ax.hist(values, bins=bins, color="#5B8FF9", alpha=0.8)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2, linewidth=0.6)

    ax.set_title(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_level_histograms(values, level_ids, title_prefix, save_path, bins, levels_per_row, discrete=False):
    plt = get_matplotlib_pyplot()
    unique_levels = [int(v) for v in np.unique(level_ids) if int(v) >= 0]

    if not unique_levels:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.text(0.5, 0.5, "No RAHT levels", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title(title_prefix)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return {}

    ncols = max(1, int(levels_per_row))
    nrows = math.ceil(len(unique_levels) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 2.8 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(-1)

    value_range = None
    if values.size > 0 and not discrete:
        min_value = float(values.min())
        max_value = float(values.max())
        if min_value != max_value:
            value_range = (-10, 10)

    level_counts = {}
    for axis, level in zip(axes, unique_levels):
        level_values = values[level_ids == level]
        level_counts[str(level)] = int(level_values.size)
        axis.set_title(f"level {level} | n={level_values.size}")

        if level_values.size == 0:
            axis.text(0.5, 0.5, "No samples", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue

        if discrete:
            unique_values, counts = np.unique(level_values.astype(np.int64), return_counts=True)
            axis.bar(unique_values, counts, width=0.9, color="#D94841", alpha=0.85)
            axis.set_xticks(select_tick_positions(unique_values))
            axis.tick_params(axis="x", labelrotation=45)
            axis.set_xlabel("Quantized symbol")
        else:
            level_min = float(level_values.min())
            level_max = float(level_values.max())
            axis.hist(level_values, bins=bins, range=(level_min, level_max), color="#5B8FF9", alpha=0.8)
            axis.set_xlabel("Coefficient value")

        axis.set_ylabel("Count")
        axis.grid(alpha=0.2, linewidth=0.6)

    for axis in axes[len(unique_levels):]:
        axis.set_axis_off()

    fig.suptitle(title_prefix)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return level_counts


def collect_raht_statistics(args):
    ply_format = infer_ply_format(args.ply_path) if args.ply_format == "auto" else args.ply_format
    importance = load_importance(args.importance)
    bit_config = build_bit_config(args)

    gaussians = build_model(args, ply_format)
    per_channel_quant, per_block_quant, effective_n_block = get_quantization_flags(args)

    gaussians.octree_coding(
        importance,
        args.oct_merge,
        raht=True,
    )
    gaussians.init_qas(
        effective_n_block,
        bit_config=bit_config,
        quant_type=args.quant_type,
        encode="deflate",
    )

    raht_input = build_raht_input_matrix(gaussians)
    raht_coeffs = apply_raht_transform(gaussians, raht_input)
    warmup_quantizers_from_raht(gaussians, raht_coeffs, per_channel_quant, per_block_quant)
    qci, dim_bits, dim_ranges = quantize_raht_ac(gaussians, raht_coeffs, per_block_quant)
    level_ids = np.asarray(getattr(gaussians, "raht_level_ids", np.zeros((0,), dtype=np.int64)))

    result = {
        "gaussians": gaussians,
        "ply_format": ply_format,
        "bit_config": bit_config,
        "effective_n_block": int(effective_n_block),
        "raht_input": raht_input.detach().cpu().numpy(),
        "raht_coeffs": raht_coeffs.detach().cpu().numpy(),
        "quantized_coeffs": qci.astype(np.int32),
        "level_ids": level_ids.astype(np.int64),
        "dim_bits": [int(bit) for bit in dim_bits],
        "dim_ranges": [[int(lo), int(hi)] for lo, hi in dim_ranges],
    }
    return result


def write_channel_plots(stats, args, output_dir):
    gaussians = stats["gaussians"]
    before_dir = output_dir / "RAHT_before"
    after_dir = output_dir / "RAHT_after"
    quantized_dir = output_dir / "quantized_after"
    channel_records = get_channel_records(gaussians)

    before_values = stats["raht_input"]
    after_values = stats["raht_coeffs"][1:, :]
    quantized_values = stats["quantized_coeffs"]
    level_ids = stats["level_ids"]

    manifest = {
        "ply_format": stats["ply_format"],
        "effective_n_block": stats["effective_n_block"],
        "level_ids_present": [int(v) for v in np.unique(level_ids) if int(v) >= 0],
        "root_dc_excluded_for_level_plots": True,
        "folders": {
            "RAHT_before": str(before_dir.resolve()),
            "RAHT_after": str(after_dir.resolve()),
            "quantized_after": str(quantized_dir.resolve()),
        },
        "channels": [],
    }

    for record in channel_records:
        attr_group = record["attr_group"]
        dim_idx = record["global_dim"]
        local_channel_index = record["local_channel_index"]
        base_name = f"{sanitize_name(attr_group)}_c{local_channel_index:02d}_dim{dim_idx:02d}.png"

        before_path = before_dir / attr_group / base_name
        after_path = after_dir / attr_group / base_name
        quantized_path = quantized_dir / attr_group / base_name

        plot_histogram(
            before_values[:, dim_idx],
            title=f"{attr_group} channel {local_channel_index} | before RAHT | dim {dim_idx}",
            save_path=before_path,
            bins=args.hist_bins,
        )
        after_level_counts = plot_level_histograms(
            after_values[:, dim_idx],
            level_ids,
            title_prefix=f"{attr_group} channel {local_channel_index} | after RAHT | dim {dim_idx}",
            save_path=after_path,
            bins=args.hist_bins,
            levels_per_row=args.levels_per_row,
            discrete=False,
        )
        quantized_level_counts = plot_level_histograms(
            quantized_values[:, dim_idx],
            level_ids,
            title_prefix=f"{attr_group} channel {local_channel_index} | after quantization | dim {dim_idx}",
            save_path=quantized_path,
            bins=args.hist_bins,
            levels_per_row=args.levels_per_row,
            discrete=True,
        )

        manifest["channels"].append(
            {
                "attr_group": attr_group,
                "global_dim": int(dim_idx),
                "local_channel_index": int(local_channel_index),
                "before_raht_plot": str(before_path.resolve()),
                "after_raht_plot": str(after_path.resolve()),
                "quantized_plot": str(quantized_path.resolve()),
                "bitwidth": int(stats["dim_bits"][dim_idx]),
                "quant_range": stats["dim_ranges"][dim_idx],
                "after_raht_level_counts": after_level_counts,
                "quantized_level_counts": quantized_level_counts,
            }
        )

    return manifest


def main():
    args = parse_args()
    if not args.ply_path.exists():
        raise FileNotFoundError(args.ply_path)
    if args.oct_merge == "imp" and args.importance is None:
        raise ValueError("--oct-merge=imp requires --importance.")
    if args.importance is not None and not args.importance.exists():
        raise FileNotFoundError(args.importance)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "outputs" / "raht_channel_histograms" / args.ply_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input PLY      : {args.ply_path}")
    print(f"Output dir     : {output_dir}")
    print(f"RAHT depth     : {args.depth}")
    print(f"Quant type     : {args.quant_type}")

    with torch.no_grad():
        stats = collect_raht_statistics(args)
        manifest = write_channel_plots(stats, args, output_dir)

    summary = {
        "ply_path": str(args.ply_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "ply_format": stats["ply_format"],
        "quant_type": args.quant_type,
        "quant_granularity": args.quant_granularity,
        "bit_config": stats["bit_config"],
        "effective_n_block": stats["effective_n_block"],
        "total_channels": len(manifest["channels"]),
        "level_ids_present": manifest["level_ids_present"],
        "root_dc_excluded_for_level_plots": True,
        "folders": manifest["folders"],
        "channels": manifest["channels"],
    }

    summary_path = output_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Total channels : {summary['total_channels']}")
    print(f"RAHT_before    : {summary['folders']['RAHT_before']}")
    print(f"RAHT_after     : {summary['folders']['RAHT_after']}")
    print(f"quantized_after: {summary['folders']['quantized_after']}")
    print(f"Summary JSON   : {summary_path}")


if __name__ == "__main__":
    main()
