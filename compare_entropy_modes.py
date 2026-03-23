import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from plyfile import PlyData
from torch import nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.gaussian_model import (
    ATTR_GROUP_ORDER,
    GaussianModel,
    ToEulerAngles_FT,
    split_length,
    transform_batched_torch,
)


DEFAULT_BIT_CONFIG = {
    "opacity": 8,
    "euler": 8,
    "scale": 10,
    "f_dc": 8,
    "f_rest_0": 4,
    "f_rest_1": 4,
    "f_rest_2": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one-shot RAHT + quantization + entropy coding and compare ANS vs Deflate."
    )
    parser.add_argument("ply_path", type=Path, help="Input Gaussian PLY file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used to store per-mode compression outputs and the summary JSON.",
    )
    parser.add_argument(
        "--ply-format",
        choices=["auto", "quat", "euler"],
        default="auto",
        help="Input PLY rotation format.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["deflate", "ans"],
        default=["deflate", "ans"],
        help="Entropy coding modes to run.",
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
    parser.add_argument(
        "--bit-packing",
        dest="bit_packing",
        action="store_true",
        default=True,
        help="Enable existing bit-packing path for deflate mode.",
    )
    parser.add_argument(
        "--no-bit-packing",
        dest="bit_packing",
        action="store_false",
        help="Disable bit packing and keep grouped storage before zip deflate.",
    )
    parser.add_argument("--ans-subgroup-count", type=int, default=4)
    parser.add_argument("--opacity-bits", type=int, default=DEFAULT_BIT_CONFIG["opacity"])
    parser.add_argument("--euler-bits", type=int, default=DEFAULT_BIT_CONFIG["euler"])
    parser.add_argument("--scale-bits", type=int, default=DEFAULT_BIT_CONFIG["scale"])
    parser.add_argument("--f-dc-bits", type=int, default=DEFAULT_BIT_CONFIG["f_dc"])
    parser.add_argument("--f-rest-0-bits", type=int, default=DEFAULT_BIT_CONFIG["f_rest_0"])
    parser.add_argument("--f-rest-1-bits", type=int, default=DEFAULT_BIT_CONFIG["f_rest_1"])
    parser.add_argument("--f-rest-2-bits", type=int, default=DEFAULT_BIT_CONFIG["f_rest_2"])
    parser.add_argument(
        "--summary-name",
        default="comparison_summary.json",
        help="Filename for the JSON summary under --output-dir.",
    )
    return parser.parse_args()


def build_bit_config(args):
    return {
        "opacity": args.opacity_bits,
        "euler": args.euler_bits,
        "scale": args.scale_bits,
        "f_dc": args.f_dc_bits,
        "f_rest_0": args.f_rest_0_bits,
        "f_rest_1": args.f_rest_1_bits,
        "f_rest_2": args.f_rest_2_bits,
    }


def infer_ply_format(ply_path):
    plydata = PlyData.read(str(ply_path))
    prop_names = {prop.name for prop in plydata.elements[0].properties}
    if any(name.startswith("euler_") for name in prop_names):
        return "euler"
    if any(name.startswith("rot") for name in prop_names):
        return "quat"
    raise ValueError(f"Unable to infer rotation format from {ply_path}.")


def load_importance(path):
    if path is None:
        return None

    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".npz":
        npz = np.load(path)
        if "imp" in npz:
            data = npz["imp"]
        else:
            first_key = npz.files[0]
            data = npz[first_key]
    else:
        raise ValueError(f"Unsupported importance file: {path}")

    return np.asarray(data).reshape(-1)


def euler_to_quaternion(eulers):
    roll = eulers[:, 0]
    pitch = eulers[:, 1]
    yaw = eulers[:, 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=1)


def build_model(args, ply_format):
    gaussians = GaussianModel(args.sh_degree, depth=args.depth, num_bits=args.num_bits)
    if ply_format == "quat":
        gaussians.load_ply(str(args.ply_path))
    else:
        gaussians.load_ply_euler(str(args.ply_path))
        quaternions = euler_to_quaternion(gaussians.get_euler.detach())
        gaussians._rotation = nn.Parameter(quaternions.requires_grad_(False))
    return gaussians


def get_quantization_flags(args):
    if args.quant_granularity == "per_channel":
        return True, False, 1
    return False, True, args.n_block


def update_ans_models(gaussians):
    if not hasattr(gaussians, "ans_entropy_bottlenecks"):
        return
    for eb in gaussians.ans_entropy_bottlenecks.values():
        eb.update(force=True)


def collect_file_sizes(mode_dir):
    bins_dir = mode_dir / "bins"
    file_sizes = {}
    for path in sorted(bins_dir.glob("*")):
        if path.is_file():
            file_sizes[path.name] = path.stat().st_size

    raw_bins_bytes = sum(file_sizes.values())
    zip_path = mode_dir / "bins.zip"
    zip_bytes = zip_path.stat().st_size if zip_path.exists() else 0

    return {
        "files": file_sizes,
        "raw_bins_bytes": raw_bins_bytes,
        "zip_bytes": zip_bytes,
        "orgb_bytes": file_sizes.get("orgb.npz", 0),
        "oct_bytes": file_sizes.get("oct.npz", 0),
        "t_bytes": file_sizes.get("t.npz", 0),
    }


def ratio(num, den):
    if den == 0:
        return None
    return num / den


def format_ratio(num, den):
    value = ratio(num, den)
    if value is None:
        return "n/a"
    return f"{value:.6f}x"


def build_raht_feature_matrix(gaussians):
    r = gaussians.get_ori_rotation
    norm = torch.sqrt(torch.sum(r * r, dim=1))
    q = r / norm[:, None]
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

    C = rf[gaussians.reorder].clone()
    iW1 = gaussians.res["iW1"]
    iW2 = gaussians.res["iW2"]
    iLeft_idx = gaussians.res["iLeft_idx"]
    iRight_idx = gaussians.res["iRight_idx"]

    for depth_idx in range(gaussians.depth * 3):
        w1 = iW1[depth_idx]
        w2 = iW2[depth_idx]
        left_idx = iLeft_idx[depth_idx]
        right_idx = iRight_idx[depth_idx]
        C[left_idx], C[right_idx] = transform_batched_torch(
            w1,
            w2,
            C[left_idx],
            C[right_idx],
        )
    return C


def estimate_distribution_from_entropy_model(eb, support_values):
    if not support_values:
        return {}, 0.0

    device = eb.quantiles.device
    was_training = eb.training
    eb.eval()
    with torch.no_grad():
        x = torch.tensor(support_values, dtype=torch.float32, device=device).view(1, 1, -1, 1)
        _, likelihood = eb(x)
        likelihood = likelihood.reshape(-1).detach().cpu().numpy().astype(np.float64)

    if was_training:
        eb.train()

    support_mass = float(likelihood.sum())
    if support_mass > 0:
        normalized = likelihood / support_mass
    else:
        normalized = np.zeros_like(likelihood)

    probabilities = {
        str(int(symbol)): float(prob)
        for symbol, prob in zip(support_values, normalized.tolist())
    }
    return probabilities, support_mass


def quantize_raht_ac(gaussians, C, per_block_quant):
    ac_len = C.shape[0] - 1
    split = split_length(ac_len, gaussians.n_block) if per_block_quant else [ac_len]
    qci = []
    dim_bits = []
    dim_ranges = []
    qa_cnt = 0

    for dim_idx in range(C.shape[-1]):
        qas_for_dim = gaussians.qas[qa_cnt : qa_cnt + len(split)]
        q_dim, _ = gaussians.quantize_ac_dimension_for_ans(C[1:, dim_idx], qas_for_dim, split)
        qci.append(q_dim.reshape(-1, 1))
        dim_bits.append(int(qas_for_dim[0].bit))
        dim_ranges.append((int(qas_for_dim[0].thd_neg), int(qas_for_dim[0].thd_pos)))
        qa_cnt += len(split)

    qci = np.concatenate(qci, axis=-1).astype(np.int32)
    return qci, dim_bits, dim_ranges


def build_group_diagnostics(gaussians, qci, dim_bits, dim_ranges):
    group_data = gaussians.build_ans_group_tensors_from_qci(qci.astype(np.float32))
    row_subgroups = np.asarray(getattr(gaussians, "raht_subgroup_ids", np.zeros((0,), dtype=np.int64)))

    diagnostics = {
        "effective_subgroup_count": int(gaussians.ans_effective_subgroup_count),
        "total_ac_rows": int(qci.shape[0]),
        "total_ac_symbols": int(qci.size),
        "groups": {},
        "aggregated_groups": {},
        "totals": {
            "fixed_length_bits": 0,
            "empirical_entropy_bits": 0.0,
            "ans_theoretical_gain_upper_bound_bits": 0.0,
        },
    }

    for attr_group in ATTR_GROUP_ORDER:
        dims = gaussians.get_attr_group_dims(attr_group)
        dim_bitwidths = [int(dim_bits[idx]) for idx in dims]
        group_support_min = min(dim_ranges[idx][0] for idx in dims)
        group_support_max = max(dim_ranges[idx][1] for idx in dims)
        group_support = list(range(group_support_min, group_support_max + 1))
        diagnostics["groups"][attr_group] = {}

        for subgroup_idx in range(gaussians.ans_effective_subgroup_count):
            group_key = gaussians.get_ans_group_key(attr_group, subgroup_idx)
            row_mask = row_subgroups == subgroup_idx
            row_count = int(row_mask.sum())
            symbols_list = group_data[group_key]

            if symbols_list:
                symbols = torch.cat(symbols_list).detach().cpu().numpy().astype(np.int32)
                unique_vals, unique_counts = np.unique(symbols, return_counts=True)
                probabilities = unique_counts.astype(np.float64) / float(symbols.size)
                empirical_entropy = float(-(probabilities * np.log2(probabilities)).sum())
                histogram_counts = {
                    str(int(symbol)): int(count)
                    for symbol, count in zip(unique_vals.tolist(), unique_counts.tolist())
                }
                empirical_probabilities = {
                    str(int(symbol)): float(prob)
                    for symbol, prob in zip(unique_vals.tolist(), probabilities.tolist())
                }
            else:
                symbols = np.zeros((0,), dtype=np.int32)
                empirical_entropy = 0.0
                histogram_counts = {}
                empirical_probabilities = {}

            eb = gaussians.get_ans_entropy_model(group_key)
            estimated_probabilities, support_mass = estimate_distribution_from_entropy_model(
                eb,
                group_support,
            )

            fixed_length_bits = int(row_count * sum(dim_bitwidths))
            empirical_total_bits = float(empirical_entropy * symbols.size)
            ans_gain_upper_bound_bits = max(0.0, fixed_length_bits - empirical_total_bits)

            diagnostics["groups"][attr_group][f"sg{subgroup_idx}"] = {
                "group_key": group_key,
                "subgroup_index": int(subgroup_idx),
                "row_count": row_count,
                "dim_count": int(len(dims)),
                "dim_indices": [int(idx) for idx in dims],
                "bitwidths": dim_bitwidths,
                "support_min": int(group_support_min),
                "support_max": int(group_support_max),
                "support_values": [int(v) for v in group_support],
                "symbol_count": int(symbols.size),
                "histogram_counts": histogram_counts,
                "empirical_probabilities": empirical_probabilities,
                "estimated_probabilities": estimated_probabilities,
                "estimated_probability_mass_on_support": support_mass,
                "empirical_entropy_bits_per_symbol": empirical_entropy,
                "empirical_entropy_bits_total": empirical_total_bits,
                "fixed_length_bits_total": fixed_length_bits,
                "fixed_length_bits_per_symbol": (
                    float(fixed_length_bits) / float(symbols.size) if symbols.size else 0.0
                ),
                "ans_theoretical_size_bits": empirical_total_bits,
                "ans_theoretical_gain_upper_bound_bits": ans_gain_upper_bound_bits,
                "ans_theoretical_gain_upper_bound_bytes": ans_gain_upper_bound_bits / 8.0,
            }

            diagnostics["totals"]["fixed_length_bits"] += fixed_length_bits
            diagnostics["totals"]["empirical_entropy_bits"] += empirical_total_bits
            diagnostics["totals"]["ans_theoretical_gain_upper_bound_bits"] += (
                ans_gain_upper_bound_bits
            )

        aggregated_counts = {}
        aggregated_empirical_probs = {}
        aggregated_estimated_probs = {str(v): 0.0 for v in group_support}
        total_group_symbols = 0

        for subgroup_name, subgroup_info in diagnostics["groups"][attr_group].items():
            total_group_symbols += subgroup_info["symbol_count"]

            for symbol, count in subgroup_info["histogram_counts"].items():
                aggregated_counts[symbol] = aggregated_counts.get(symbol, 0) + count

            weight = subgroup_info["symbol_count"]
            if weight > 0:
                for symbol, prob in subgroup_info["estimated_probabilities"].items():
                    aggregated_estimated_probs[symbol] = (
                        aggregated_estimated_probs.get(symbol, 0.0)
                        + prob * weight
                    )

        if total_group_symbols > 0:
            aggregated_empirical_probs = {
                symbol: float(count) / float(total_group_symbols)
                for symbol, count in aggregated_counts.items()
            }
            aggregated_estimated_probs = {
                symbol: float(prob) / float(total_group_symbols)
                for symbol, prob in aggregated_estimated_probs.items()
            }
        else:
            aggregated_estimated_probs = {symbol: 0.0 for symbol in aggregated_estimated_probs}

        diagnostics["aggregated_groups"][attr_group] = {
            "bitwidths": dim_bitwidths,
            "dim_indices": [int(idx) for idx in dims],
            "support_min": int(group_support_min),
            "support_max": int(group_support_max),
            "support_values": [int(v) for v in group_support],
            "symbol_count": int(total_group_symbols),
            "histogram_counts": aggregated_counts,
            "empirical_probabilities": aggregated_empirical_probs,
            "estimated_probabilities": aggregated_estimated_probs,
        }

    diagnostics["totals"]["fixed_length_bytes"] = diagnostics["totals"]["fixed_length_bits"] / 8.0
    diagnostics["totals"]["empirical_entropy_bytes"] = diagnostics["totals"]["empirical_entropy_bits"] / 8.0
    diagnostics["totals"]["ans_theoretical_gain_upper_bound_bytes"] = (
        diagnostics["totals"]["ans_theoretical_gain_upper_bound_bits"] / 8.0
    )
    return diagnostics


def print_group_diagnostics(diagnostics):
    print("\n===== Symbol Diagnostics =====")
    print(
        "Group/Subgroup | Symbols | H(bits/sym) | Fixed(bits/sym) | "
        "Fixed(MB) | H(MB) | ANS Gain Max(MB)"
    )

    for attr_group in ATTR_GROUP_ORDER:
        for subgroup_name, info in diagnostics["groups"][attr_group].items():
            if info["symbol_count"] == 0:
                continue
            print(
                f"{attr_group:9s}/{subgroup_name:4s} | "
                f"{info['symbol_count']:8d} | "
                f"{info['empirical_entropy_bits_per_symbol']:11.4f} | "
                f"{info['fixed_length_bits_per_symbol']:15.4f} | "
                f"{info['fixed_length_bits_total'] / 8.0 / 1024 / 1024:9.4f} | "
                f"{info['empirical_entropy_bits_total'] / 8.0 / 1024 / 1024:6.4f} | "
                f"{info['ans_theoretical_gain_upper_bound_bits'] / 8.0 / 1024 / 1024:15.4f}"
            )

    totals = diagnostics["totals"]
    print(
        "TOTAL          | "
        f"{diagnostics['total_ac_symbols']:8d} | "
        f"{(totals['empirical_entropy_bits'] / max(diagnostics['total_ac_symbols'], 1)):11.4f} | "
        f"{(totals['fixed_length_bits'] / max(diagnostics['total_ac_symbols'], 1)):15.4f} | "
        f"{totals['fixed_length_bytes'] / 1024 / 1024:9.4f} | "
        f"{totals['empirical_entropy_bytes'] / 1024 / 1024:6.4f} | "
        f"{totals['ans_theoretical_gain_upper_bound_bytes'] / 1024 / 1024:15.4f}"
    )


def collect_symbol_diagnostics(args, ply_format, importance, bit_config):
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
        encode="ans",
        ans_subgroup_count=args.ans_subgroup_count,
    )
    update_ans_models(gaussians)

    C = build_raht_feature_matrix(gaussians)
    qci, dim_bits, dim_ranges = quantize_raht_ac(gaussians, C, per_block_quant)
    diagnostics = build_group_diagnostics(gaussians, qci, dim_bits, dim_ranges)
    diagnostics["dim_bitwidths"] = [int(bit) for bit in dim_bits]
    diagnostics["dim_ranges"] = [[int(lo), int(hi)] for lo, hi in dim_ranges]
    diagnostics["effective_n_block"] = int(effective_n_block)
    diagnostics["quant_granularity"] = args.quant_granularity

    del gaussians
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return diagnostics


def print_probability_distribution_summary(diagnostics):
    print("\n===== Probability Distributions =====")
    for attr_group in ATTR_GROUP_ORDER:
        info = diagnostics["aggregated_groups"][attr_group]
        print(
            f"{attr_group:9s} | symbols={info['symbol_count']:8d} | "
            f"support=[{info['support_min']}, {info['support_max']}] | "
            f"hist_bins={len(info['histogram_counts'])}"
        )


def run_single_mode(args, mode, ply_format, importance, bit_config, output_dir):
    print(f"\n===== Running mode: {mode} =====")
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
        encode=mode,
        ans_subgroup_count=args.ans_subgroup_count,
    )
    if mode == "ans":
        update_ans_models(gaussians)

    pipe_stub = SimpleNamespace()
    gaussians.save_npz(
        str(output_dir),
        pipe_stub,
        per_channel_quant=per_channel_quant,
        per_block_quant=per_block_quant,
        bit_packing=args.bit_packing,
    )

    sizes = collect_file_sizes(output_dir)
    point_count = int(gaussians.get_xyz.shape[0])
    oct_point_count = int(gaussians.oct.shape[0]) if hasattr(gaussians, "oct") else point_count

    result = {
        "mode": mode,
        "point_count_after_octree": oct_point_count,
        "decoded_point_count": point_count,
        "effective_n_block": effective_n_block,
        "quant_granularity": args.quant_granularity,
        "bit_packing": bool(args.bit_packing),
        "ans_subgroup_count": int(args.ans_subgroup_count),
    }
    result.update(sizes)

    del gaussians
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def print_summary(results, original_bytes):
    mb = 1024.0 * 1024.0
    print("\n===== Compression Summary =====")
    print(f"Original PLY: {original_bytes / mb:.4f} MB ({original_bytes} bytes)")
    for mode, info in results.items():
        print(f"\n[{mode}]")
        print(
            f"  orgb.npz   : {info['orgb_bytes'] / mb:.4f} MB | ratio vs PLY = "
            f"{format_ratio(original_bytes, info['orgb_bytes'])}"
        )
        print(
            f"  bins raw   : {info['raw_bins_bytes'] / mb:.4f} MB | ratio vs PLY = "
            f"{format_ratio(original_bytes, info['raw_bins_bytes'])}"
        )
        print(
            f"  bins.zip   : {info['zip_bytes'] / mb:.4f} MB | ratio vs PLY = "
            f"{format_ratio(original_bytes, info['zip_bytes'])}"
        )
        print(
            f"  oct / t    : {info['oct_bytes'] / mb:.4f} MB / {info['t_bytes'] / mb:.4f} MB"
        )

    if "ans" in results and "deflate" in results:
        ans = results["ans"]
        deflate = results["deflate"]
        print("\n[Delta]")
        print(
            f"  orgb bytes : ans / deflate = "
            f"{format_ratio(ans['orgb_bytes'], deflate['orgb_bytes'])}"
        )
        print(
            f"  zip bytes  : ans / deflate = "
            f"{format_ratio(ans['zip_bytes'], deflate['zip_bytes'])}"
        )


def main():
    args = parse_args()
    if not args.ply_path.exists():
        raise FileNotFoundError(args.ply_path)
    if args.oct_merge == "imp" and args.importance is None:
        raise ValueError("--oct-merge=imp requires --importance.")
    if args.importance is not None and not args.importance.exists():
        raise FileNotFoundError(args.importance)

    ply_format = infer_ply_format(args.ply_path) if args.ply_format == "auto" else args.ply_format
    importance = load_importance(args.importance)
    bit_config = build_bit_config(args)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "outputs" / "entropy_compare" / args.ply_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    original_bytes = args.ply_path.stat().st_size
    results = {}

    print(f"Input PLY      : {args.ply_path}")
    print(f"PLY format     : {ply_format}")
    print(f"Output dir     : {output_dir}")
    print(f"Modes          : {args.modes}")
    print(f"RAHT depth     : {args.depth}")
    print(f"Octree merge   : {args.oct_merge}")
    print(f"Quant type     : {args.quant_type}")
    print(f"Bit config     : {bit_config}")

    with torch.no_grad():
        symbol_diagnostics = collect_symbol_diagnostics(
            args,
            ply_format,
            importance,
            bit_config,
        )

        for mode in args.modes:
            mode_dir = output_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            results[mode] = run_single_mode(
                args,
                mode,
                ply_format,
                importance,
                bit_config,
                mode_dir,
            )

    diagnostics_path = output_dir / "symbol_diagnostics.json"
    diagnostics_path.write_text(json.dumps(symbol_diagnostics, indent=2), encoding="utf-8")

    summary = {
        "ply_path": str(args.ply_path.resolve()),
        "ply_format": ply_format,
        "original_ply_bytes": original_bytes,
        "oct_merge": args.oct_merge,
        "quant_type": args.quant_type,
        "quant_granularity": args.quant_granularity,
        "bit_packing": bool(args.bit_packing),
        "bit_config": bit_config,
        "ans_subgroup_count": int(args.ans_subgroup_count),
        "symbol_diagnostics_path": str(diagnostics_path.resolve()),
        "symbol_diagnostics_totals": symbol_diagnostics["totals"],
        "results": results,
    }

    summary_path = output_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_probability_distribution_summary(symbol_diagnostics)
    print_group_diagnostics(symbol_diagnostics)
    print_summary(results, original_bytes)
    print(f"Symbol diagnostics JSON: {diagnostics_path}")
    print(f"\nSummary JSON: {summary_path}")


if __name__ == "__main__":
    main()
