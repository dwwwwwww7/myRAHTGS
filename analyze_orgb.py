import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from compressai.entropy_models import EntropyBottleneck

from scene.gaussian_model import ATTR_GROUP_SLICES, unpack_bits


ATTR_TITLES = {
    "opacity": "Opacity",
    "euler": "Euler Angles",
    "f_dc": "Features DC",
    "f_rest_0": "Features Rest (SH Deg 1)",
    "f_rest_1": "Features Rest (SH Deg 2)",
    "f_rest_2": "Features Rest (SH Deg 3)",
    "scale": "Scale",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a single orgb.npz and draw quantized symbol histograms."
    )
    parser.add_argument(
        "orgb_path",
        type=Path,
        help="Path to orgb.npz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output png path. Defaults next to orgb.npz.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure after saving.",
    )
    return parser.parse_args()


def resolve_output_path(orgb_path: Path, output: Optional[Path]) -> Path:
    if output is not None:
        return output
    return orgb_path.with_name(f"{orgb_path.stem}_distribution.png")


def infer_storage_mode(data):
    if "packed" in data and int(data["packed"][0]) == 2:
        return "ans"
    if "packed" in data and int(data["packed"][0]) == 1:
        return "bit_packed"
    if any(key.startswith("i_") and key.endswith("bit") for key in data.files):
        return "grouped"
    if "i" in data:
        return "dense"
    return "unknown"


def decode_bit_packed(data):
    bitstream = data["i"].tobytes()
    bit_config = data["bit_config"].astype(np.int32).tolist()
    signed_config = (
        data["signed_config"].astype(bool).tolist()
        if "signed_config" in data
        else [False] * len(bit_config)
    )

    bits_per_row = int(sum(bit_config))
    if bits_per_row <= 0:
        raise ValueError("Invalid bit_config: total bits per row must be positive.")

    row_count = (len(bitstream) * 8) // bits_per_row
    qci = unpack_bits(bitstream, bit_config, row_count, signed_flags=signed_config)
    return qci.astype(np.float32)


def decode_grouped(data):
    qci = None
    for key in data.files:
        if not (key.startswith("i_") and key.endswith("bit")):
            continue

        bit = int(key.split("_")[1].replace("bit", ""))
        dims_key = f"dims_{bit}bit"
        if dims_key not in data:
            continue

        group_data = data[key].astype(np.float32)
        dims = data[dims_key].astype(np.int32).tolist()
        if qci is None:
            qci = np.zeros((group_data.shape[0], 55), dtype=np.float32)
        qci[:, dims] = group_data

    if qci is None:
        raise ValueError("Grouped storage detected but no valid i_*bit payload was found.")
    return qci


def decode_dense(data):
    qci = data["i"].astype(np.float32)
    if qci.ndim == 1:
        if qci.size % 55 != 0:
            raise ValueError(f"Dense payload length {qci.size} is not divisible by 55.")
        qci = qci.reshape(-1, 55)
    elif qci.ndim == 2 and qci.shape[1] != 55 and qci.shape[0] == 55:
        qci = qci.T
    elif qci.ndim != 2:
        raise ValueError(f"Unsupported dense payload shape: {qci.shape}")
    return qci


def restore_ans_entropy_model(group_key, data):
    eb = EntropyBottleneck(1)
    with torch.no_grad():
        eb.quantiles.copy_(torch.tensor(data[f"quantiles_{group_key}"], dtype=torch.float32))
    eb._quantized_cdf = torch.tensor(data[f"cdf_{group_key}"], dtype=torch.int32)
    eb._cdf_length = torch.tensor(data[f"cdf_length_{group_key}"], dtype=torch.int32)
    eb._offset = torch.tensor(data[f"offset_{group_key}"], dtype=torch.int32)
    eb.eval()
    return eb


def decode_ans_group_symbols(data):
    grouped = {}
    subgroup_count = int(data["ans_subgroup_count"][0]) if "ans_subgroup_count" in data else 1

    for attr_group in ATTR_GROUP_SLICES:
        all_symbols = []
        subgroup_info = []

        for subgroup_idx in range(subgroup_count):
            group_key = f"{attr_group}__sg{subgroup_idx}"
            ans_key = f"ans_{group_key}"
            shape_key = f"shape_{group_key}"
            if ans_key not in data or shape_key not in data:
                continue

            eb = restore_ans_entropy_model(group_key, data)
            bitstream = bytes(data[ans_key].tolist())
            shape = tuple(int(v) for v in data[shape_key].tolist())
            symbols = eb.decompress([bitstream], shape[2:]).reshape(-1).cpu().numpy().astype(np.float32)
            all_symbols.append(symbols)
            subgroup_info.append((group_key, symbols))

        if all_symbols:
            grouped[attr_group] = {
                "symbols": np.concatenate(all_symbols, axis=0),
                "subgroups": subgroup_info,
            }
        else:
            grouped[attr_group] = {
                "symbols": np.zeros((0,), dtype=np.float32),
                "subgroups": [],
            }

    return grouped


def print_basic_info(orgb_path, data, mode):
    print("=" * 60)
    print(f"Analyzing: {orgb_path}")
    print(f"Storage mode: {mode}")
    print("=" * 60)
    print("Keys:", data.files)
    if "f" in data:
        print(f"DC coefficients shape: {data['f'].shape}")


def print_non_ans_summary(qci):
    print(f"Decoded AC matrix shape: {qci.shape}")
    print("\nPer-attribute ranges:")
    for attr_group, slc in ATTR_GROUP_SLICES.items():
        vals = qci[:, slc].reshape(-1)
        print(
            f"  {ATTR_TITLES[attr_group]:<24s} "
            f"min={vals.min():>8.1f} max={vals.max():>8.1f} "
            f"mean={vals.mean():>10.4f} std={vals.std():>10.4f}"
        )


def print_ans_summary(grouped_symbols, data):
    total_bytes = 0
    print("\nANS group summary:")
    for attr_group, info in grouped_symbols.items():
        symbols = info["symbols"]
        print(
            f"  {ATTR_TITLES[attr_group]:<24s} "
            f"symbols={symbols.size:>8d} "
            f"min={symbols.min() if symbols.size else 'n/a'} "
            f"max={symbols.max() if symbols.size else 'n/a'}"
        )
        for group_key, subgroup_symbols in info["subgroups"]:
            ans_key = f"ans_{group_key}"
            byte_count = int(data[ans_key].size)
            total_bytes += byte_count
            print(
                f"    {group_key:<24s} bytes={byte_count:>8d} "
                f"symbols={subgroup_symbols.size:>8d}"
            )
    print(f"\nTotal ANS bytes: {total_bytes}")


def plot_non_ans_histograms(qci, output_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle(f"Quantized AC Coefficients Distribution\n{output_path.stem}", fontsize=16, y=0.98)

    for idx, (attr_group, slc) in enumerate(ATTR_GROUP_SLICES.items()):
        ax = axes[idx]
        vals = qci[:, slc].reshape(-1)
        ax.hist(vals, bins=50, alpha=0.75, color="steelblue", edgecolor="black", log=True)
        ax.set_title(f"{ATTR_TITLES[attr_group]} (Dims {slc.start}-{slc.stop - 1})")
        ax.set_xlabel("Quantized Value")
        ax.set_ylabel("Frequency (Log Scale)")
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(ATTR_GROUP_SLICES), len(axes)):
        fig.delaxes(axes[idx])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_ans_histograms(grouped_symbols, output_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle(f"ANS Decoded Symbol Distribution\n{output_path.stem}", fontsize=16, y=0.98)

    for idx, attr_group in enumerate(ATTR_GROUP_SLICES):
        ax = axes[idx]
        vals = grouped_symbols[attr_group]["symbols"]
        if vals.size > 0:
            unique_vals = np.unique(vals)
            bins = min(max(unique_vals.size, 10), 80)
            ax.hist(vals, bins=bins, alpha=0.75, color="steelblue", edgecolor="black", log=True)
        ax.set_title(ATTR_TITLES[attr_group])
        ax.set_xlabel("Quantized Symbol")
        ax.set_ylabel("Frequency (Log Scale)")
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(ATTR_GROUP_SLICES), len(axes)):
        fig.delaxes(axes[idx])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    orgb_path = args.orgb_path
    if not orgb_path.exists():
        raise FileNotFoundError(orgb_path)

    output_path = resolve_output_path(orgb_path, args.output)
    data = np.load(orgb_path, allow_pickle=False)
    mode = infer_storage_mode(data)
    print_basic_info(orgb_path, data, mode)

    if mode == "bit_packed":
        qci = decode_bit_packed(data)
        print_non_ans_summary(qci)
        plot_non_ans_histograms(qci, output_path)
    elif mode == "grouped":
        qci = decode_grouped(data)
        print_non_ans_summary(qci)
        plot_non_ans_histograms(qci, output_path)
    elif mode == "dense":
        qci = decode_dense(data)
        print_non_ans_summary(qci)
        plot_non_ans_histograms(qci, output_path)
    elif mode == "ans":
        grouped_symbols = decode_ans_group_symbols(data)
        print_ans_summary(grouped_symbols, data)
        plot_ans_histograms(grouped_symbols, output_path)
    else:
        raise ValueError(f"Unsupported orgb.npz format: keys={data.files}")

    print(f"\nSaved histogram figure: {output_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
