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
        "--offline-fit-steps",
        type=int,
        default=100,
        help="Number of offline fitting steps for EntropyBottleneck before ANS encoding. Use 0 to disable.",
    )
    parser.add_argument(
        "--offline-fit-plot-interval",
        type=int,
        default=10,
        help="Write fitted probability plots every N offline fitting steps.",
    )
    parser.add_argument(
        "--offline-fit-main-lr",
        type=float,
        default=1e-3,
        help="Learning rate for EntropyBottleneck main parameters during offline fitting.",
    )
    parser.add_argument(
        "--offline-fit-aux-lr",
        type=float,
        default=1e-3,
        help="Learning rate for EntropyBottleneck quantiles during offline fitting.",
    )
    parser.add_argument(
        "--summary-name",
        default="comparison_summary.json",
        help="Filename for the JSON summary under --output-dir.",
    )
    parser.add_argument(
        "--skip-probability-plots",
        action="store_true",
        help="Skip writing PNG plots for empirical vs estimated probability distributions.",
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


def get_matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def prepare_probability_arrays(info):
    support_values = [int(v) for v in info["support_values"]]
    empirical = np.array(
        [float(info["empirical_probabilities"].get(str(v), 0.0)) for v in support_values],
        dtype=np.float64,
    )
    estimated = np.array(
        [float(info["estimated_probabilities"].get(str(v), 0.0)) for v in support_values],
        dtype=np.float64,
    )
    return np.asarray(support_values, dtype=np.int32), empirical, estimated


def select_tick_positions(x_values, max_ticks=17):
    if x_values.size <= max_ticks:
        return x_values
    indices = np.linspace(0, x_values.size - 1, num=max_ticks, dtype=int)
    return x_values[indices]


def sanitize_name(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def clone_module_state_to_cpu(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def filter_entropy_bottleneck_runtime_buffers(state_dict):
    ignored_suffixes = (
        "._offset",
        "._quantized_cdf",
        "._cdf_length",
    )
    return {
        key: value
        for key, value in state_dict.items()
        if not key.endswith(ignored_suffixes)
    }


def load_fitted_entropy_bottleneck_state(module, state_dict):
    filtered_state = filter_entropy_bottleneck_runtime_buffers(state_dict)
    module.load_state_dict(filtered_state, strict=False)


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


def warmup_quantizers_from_raht(gaussians, C, per_channel_quant, per_block_quant):
    if not hasattr(gaussians, "qas") or len(gaussians.qas) == 0:
        return

    if per_channel_quant:
        for dim_idx in range(C.shape[-1]):
            qa = gaussians.qas[dim_idx]
            x = C[1:, dim_idx]
            if hasattr(qa, "init_yet"):
                if not qa.init_yet:
                    qa.init_from(x)
            else:
                qa(x)
        return

    if per_block_quant:
        ac_len = C.shape[0] - 1
        split_ac = split_length(ac_len, gaussians.n_block)
        qa_cnt = 0

        for dim_idx in range(C.shape[-1]):
            start_idx = 1
            for block_idx, length in enumerate(split_ac):
                end_idx = start_idx + length
                qa = gaussians.qas[qa_cnt + block_idx]
                x = C[start_idx:end_idx, dim_idx]
                if hasattr(qa, "init_yet"):
                    if not qa.init_yet:
                        qa.init_from(x)
                else:
                    qa(x)
                start_idx = end_idx
            qa_cnt += gaussians.n_block
        return

    qa = gaussians.qa
    x = C[1:]
    if hasattr(qa, "init_yet"):
        if not qa.init_yet:
            qa.init_from(x)
    else:
        qa(x)


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


def build_group_fit_targets(gaussians, qci, dim_bits, dim_ranges):
    group_data = gaussians.build_ans_group_tensors_from_qci(qci.astype(np.float32))
    targets = {}

    total_symbols = 0
    for attr_group in ATTR_GROUP_ORDER:
        dims = gaussians.get_attr_group_dims(attr_group)
        group_support_min = min(dim_ranges[idx][0] for idx in dims)
        group_support_max = max(dim_ranges[idx][1] for idx in dims)
        support_values = list(range(group_support_min, group_support_max + 1))
        support_size = len(support_values)

        for subgroup_idx in range(gaussians.ans_effective_subgroup_count):
            group_key = gaussians.get_ans_group_key(attr_group, subgroup_idx)
            symbols_list = group_data[group_key]
            counts = np.zeros((support_size,), dtype=np.float32)
            symbol_count = 0

            if symbols_list:
                symbols = torch.cat(symbols_list).detach().cpu().numpy().astype(np.int32)
                symbol_count = int(symbols.size)
                if symbol_count > 0:
                    unique_vals, unique_counts = np.unique(symbols, return_counts=True)
                    indices = unique_vals - group_support_min
                    valid_mask = (indices >= 0) & (indices < support_size)
                    counts[indices[valid_mask]] = unique_counts[valid_mask].astype(np.float32)

            targets[group_key] = {
                "support_values": support_values,
                "support_min": int(group_support_min),
                "support_max": int(group_support_max),
                "counts": counts,
                "symbol_count": symbol_count,
            }
            total_symbols += symbol_count

    return targets, total_symbols


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


def collect_symbol_diagnostics(args, ply_format, importance, bit_config, output_dir):
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
    warmup_quantizers_from_raht(gaussians, C, per_channel_quant, per_block_quant)
    qci, dim_bits, dim_ranges = quantize_raht_ac(gaussians, C, per_block_quant)
    diagnostics, fitted_state, fit_history = offline_fit_entropy_bottlenecks(
        gaussians,
        qci,
        dim_bits,
        dim_ranges,
        args,
        output_dir,
    )
    diagnostics["dim_bitwidths"] = [int(bit) for bit in dim_bits]
    diagnostics["dim_ranges"] = [[int(lo), int(hi)] for lo, hi in dim_ranges]
    diagnostics["effective_n_block"] = int(effective_n_block)
    diagnostics["quant_granularity"] = args.quant_granularity
    diagnostics["offline_fit_steps"] = int(args.offline_fit_steps)
    diagnostics["offline_fit_plot_interval"] = int(args.offline_fit_plot_interval)
    diagnostics["offline_fit_history"] = fit_history

    del gaussians
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return diagnostics, fitted_state


def print_probability_distribution_summary(diagnostics):
    print("\n===== Probability Distributions =====")
    for attr_group in ATTR_GROUP_ORDER:
        info = diagnostics["aggregated_groups"][attr_group]
        print(
            f"{attr_group:9s} | symbols={info['symbol_count']:8d} | "
            f"support=[{info['support_min']}, {info['support_max']}] | "
            f"hist_bins={len(info['histogram_counts'])}"
        )


def plot_probability_distribution(info, title, save_path):
    plt = get_matplotlib_pyplot()
    x_values, empirical, estimated = prepare_probability_arrays(info)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 2]},
        constrained_layout=True,
    )

    axes[0].bar(
        x_values,
        empirical,
        width=0.9,
        color="#5B8FF9",
        alpha=0.55,
        label="Empirical probability",
        align="center",
    )
    axes[0].plot(
        x_values,
        estimated,
        color="#D94841",
        linewidth=1.8,
        label="CompressAI estimated probability",
    )
    axes[0].set_ylabel("Probability")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2, linewidth=0.6)

    counts = np.array(
        [int(info["histogram_counts"].get(str(v), 0)) for v in x_values],
        dtype=np.int64,
    )
    axes[1].bar(
        x_values,
        counts,
        width=0.9,
        color="#7FC8A9",
        alpha=0.9,
    )
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Quantized symbol")
    axes[1].grid(alpha=0.2, linewidth=0.6)

    tick_positions = select_tick_positions(x_values)
    axes[0].set_xticks(tick_positions)
    axes[1].set_xticks(tick_positions)
    axes[0].tick_params(axis="x", labelrotation=45)
    axes[1].tick_params(axis="x", labelrotation=45)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_probability_plots(diagnostics, output_dir):
    plots_root = output_dir / "probability_plots"
    aggregated_dir = plots_root / "aggregated"
    subgroup_dir = plots_root / "subgroups"
    manifest = {
        "aggregated": {},
        "subgroups": {},
    }

    for attr_group in ATTR_GROUP_ORDER:
        aggregated_info = diagnostics["aggregated_groups"][attr_group]
        aggregated_path = aggregated_dir / f"{sanitize_name(attr_group)}.png"
        title = (
            f"{attr_group} | empirical vs CompressAI estimated probability "
            f"(symbols={aggregated_info['symbol_count']})"
        )
        plot_probability_distribution(aggregated_info, title, aggregated_path)
        manifest["aggregated"][attr_group] = str(aggregated_path.resolve())

        manifest["subgroups"][attr_group] = {}
        for subgroup_name, subgroup_info in diagnostics["groups"][attr_group].items():
            if subgroup_info["symbol_count"] == 0:
                continue
            subgroup_path = subgroup_dir / attr_group / f"{sanitize_name(subgroup_name)}.png"
            subgroup_title = (
                f"{attr_group}/{subgroup_name} | empirical vs CompressAI estimated probability "
                f"(symbols={subgroup_info['symbol_count']})"
            )
            plot_probability_distribution(subgroup_info, subgroup_title, subgroup_path)
            manifest["subgroups"][attr_group][subgroup_name] = str(subgroup_path.resolve())

    manifest_path = plots_root / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return plots_root, manifest_path


def create_fit_progress_plots(diagnostics, output_dir, step):
    plots_root = output_dir / "probability_plots_fit_progress" / f"step_{step:04d}"
    aggregated_dir = plots_root / "aggregated"
    manifest = {"aggregated": {}}

    for attr_group in ATTR_GROUP_ORDER:
        aggregated_info = diagnostics["aggregated_groups"][attr_group]
        aggregated_path = aggregated_dir / f"{sanitize_name(attr_group)}.png"
        title = (
            f"{attr_group} | fitted step {step} | empirical vs CompressAI estimated probability "
            f"(symbols={aggregated_info['symbol_count']})"
        )
        plot_probability_distribution(aggregated_info, title, aggregated_path)
        manifest["aggregated"][attr_group] = str(aggregated_path.resolve())

    manifest_path = plots_root / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def offline_fit_entropy_bottlenecks(
    gaussians,
    qci,
    dim_bits,
    dim_ranges,
    args,
    output_dir,
):
    steps = max(0, int(args.offline_fit_steps))
    if steps == 0:
        update_ans_models(gaussians)
        diagnostics = build_group_diagnostics(gaussians, qci, dim_bits, dim_ranges)
        return diagnostics, clone_module_state_to_cpu(gaussians.ans_entropy_bottlenecks), []

    fit_targets, total_symbols = build_group_fit_targets(gaussians, qci, dim_bits, dim_ranges)
    if total_symbols <= 0:
        update_ans_models(gaussians)
        diagnostics = build_group_diagnostics(gaussians, qci, dim_bits, dim_ranges)
        return diagnostics, clone_module_state_to_cpu(gaussians.ans_entropy_bottlenecks), []

    device = next(iter(gaussians.ans_entropy_bottlenecks.parameters())).device
    prepared_targets = {}
    for group_key, target in fit_targets.items():
        prepared_targets[group_key] = {
            "symbol_count": target["symbol_count"],
            "support_tensor": torch.tensor(
                target["support_values"],
                dtype=torch.float32,
                device=device,
            ).view(1, 1, -1, 1),
            "counts_tensor": torch.tensor(
                target["counts"],
                dtype=torch.float32,
                device=device,
            ),
        }

    main_params = [
        param for name, param in gaussians.ans_entropy_bottlenecks.named_parameters()
        if not name.endswith("quantiles")
    ]
    aux_params = [eb.quantiles for eb in gaussians.ans_entropy_bottlenecks.values()]

    main_optimizer = torch.optim.Adam(main_params, lr=args.offline_fit_main_lr) if main_params else None
    aux_optimizer = torch.optim.Adam(aux_params, lr=args.offline_fit_aux_lr) if aux_params else None

    fit_history = []
    plot_interval = max(1, int(args.offline_fit_plot_interval))

    for eb in gaussians.ans_entropy_bottlenecks.values():
        eb.train()

    for step in range(1, steps + 1):
        if main_optimizer is not None:
            main_optimizer.zero_grad(set_to_none=True)

        total_bits_per_symbol = 0.0
        total_nll = 0.0

        for group_key, target in prepared_targets.items():
            if target["symbol_count"] <= 0:
                continue

            eb = gaussians.get_ans_entropy_model(group_key)
            _, likelihood = eb(target["support_tensor"])
            likelihood = likelihood.reshape(-1).clamp(min=1e-9)

            weighted_nll = -(target["counts_tensor"] * torch.log(likelihood)).sum()
            loss = weighted_nll / float(total_symbols)

            if main_optimizer is not None:
                loss.backward()

            total_nll += float(loss.detach().item())
            total_bits_per_symbol += float(
                (-(target["counts_tensor"] * torch.log2(likelihood)).sum() / float(total_symbols)).detach().item()
            )

        if main_optimizer is not None:
            main_optimizer.step()

        aux_loss_value = 0.0
        if aux_optimizer is not None:
            aux_optimizer.zero_grad(set_to_none=True)
            aux_loss = torch.tensor(0.0, device=device)
            for eb in gaussians.ans_entropy_bottlenecks.values():
                aux_loss = aux_loss + eb.loss()
            if aux_loss.requires_grad:
                aux_loss.backward()
                aux_optimizer.step()
                aux_loss_value = float(aux_loss.detach().item())

        checkpoint_manifest = None
        if step % plot_interval == 0 or step == steps:
            update_ans_models(gaussians)
            diagnostics = build_group_diagnostics(gaussians, qci, dim_bits, dim_ranges)
            if not args.skip_probability_plots:
                checkpoint_manifest = create_fit_progress_plots(diagnostics, output_dir, step)
        else:
            diagnostics = None

        fit_history.append(
            {
                "step": int(step),
                "nll_per_symbol_nats": float(total_nll),
                "bits_per_symbol": float(total_bits_per_symbol),
                "aux_loss": float(aux_loss_value),
                "plot_manifest": str(checkpoint_manifest.resolve()) if checkpoint_manifest else None,
            }
        )

        if step % plot_interval == 0 or step == steps:
            print(
                f"[offline-fit] step {step:4d}/{steps} | "
                f"bits/sym={total_bits_per_symbol:.6f} | aux={aux_loss_value:.6f}"
            )

    update_ans_models(gaussians)
    final_diagnostics = build_group_diagnostics(gaussians, qci, dim_bits, dim_ranges)
    fitted_state = clone_module_state_to_cpu(gaussians.ans_entropy_bottlenecks)
    return final_diagnostics, fitted_state, fit_history


def run_single_mode(args, mode, ply_format, importance, bit_config, output_dir, fitted_ans_state=None):
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
    C = build_raht_feature_matrix(gaussians)
    warmup_quantizers_from_raht(gaussians, C, per_channel_quant, per_block_quant)
    if mode == "ans":
        if fitted_ans_state is not None:
            load_fitted_entropy_bottleneck_state(gaussians.ans_entropy_bottlenecks, fitted_ans_state)
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

    symbol_diagnostics, fitted_ans_state = collect_symbol_diagnostics(
        args,
        ply_format,
        importance,
        bit_config,
        output_dir,
    )

    with torch.no_grad():
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
                fitted_ans_state=fitted_ans_state if mode == "ans" else None,
            )

    diagnostics_path = output_dir / "symbol_diagnostics.json"
    diagnostics_path.write_text(json.dumps(symbol_diagnostics, indent=2), encoding="utf-8")

    plot_root = None
    plot_manifest_path = None
    if not args.skip_probability_plots:
        plot_root, plot_manifest_path = create_probability_plots(symbol_diagnostics, output_dir)

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
        "probability_plots_dir": str(plot_root.resolve()) if plot_root is not None else None,
        "probability_plot_manifest": (
            str(plot_manifest_path.resolve()) if plot_manifest_path is not None else None
        ),
        "results": results,
    }

    summary_path = output_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_probability_distribution_summary(symbol_diagnostics)
    print_group_diagnostics(symbol_diagnostics)
    print_summary(results, original_bytes)
    print(f"Symbol diagnostics JSON: {diagnostics_path}")
    if plot_root is not None:
        print(f"Probability plots dir: {plot_root}")
        print(f"Probability plot manifest: {plot_manifest_path}")
    print(f"\nSummary JSON: {summary_path}")


if __name__ == "__main__":
    main()
