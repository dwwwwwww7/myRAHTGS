import math
import time

import torch
import torch.nn as nn
from torch.autograd import Function
from compressai.entropy_models import EntropyBottleneck


def split_length(length, n):
    base_length = length / n
    floor_length = int(base_length)
    remainder = length - (floor_length * n)
    result = [floor_length + 1] * remainder + [floor_length] * (n - remainder)
    return result


def _build_block_index_tensors(split, device):
    split_tensor = torch.as_tensor(split, device=device, dtype=torch.long)
    if split_tensor.numel() == 0:
        raise ValueError("split must contain at least one block")

    block_ids = torch.repeat_interleave(
        torch.arange(split_tensor.numel(), device=device, dtype=torch.long),
        split_tensor,
    )
    block_starts = torch.cumsum(
        torch.cat([split_tensor.new_zeros(1), split_tensor[:-1]]),
        dim=0,
    )
    local_ids = torch.arange(int(split_tensor.sum().item()), device=device, dtype=torch.long)
    local_ids = local_ids - torch.repeat_interleave(block_starts, split_tensor)
    return split_tensor, block_ids, local_ids


def _laplace_cdf(x, scale, mean=None):
    """Numerically stable Laplace CDF.

    Uses exp(-|x|/scale) in both branches to avoid the classic
    torch.where + exp overflow:  when x > 0 the old ``exp(x/scale)``
    overflows to Inf, and the backward of ``torch.where`` computes
    ``0 * Inf = NaN``.  Using ``exp(-|x|/scale)`` (exponent always ≤ 0)
    guarantees no overflow while remaining mathematically equivalent.
    """
    if mean is not None:
        x = x - mean
    safe_scale = scale.clamp(min=1e-6)
    abs_x = x.abs()
    exp_term = torch.exp(-abs_x / safe_scale)
    # x >= 0 : CDF = 1 - 0.5 * exp(-x/b)
    # x <  0 : CDF = 0.5 * exp(-|x|/b)
    return torch.where(x >= 0, 1.0 - 0.5 * exp_term, 0.5 * exp_term)


def _get_quantizer_zero_point_value(qa):
    if hasattr(qa, "get_effective_zero_point"):
        zero_point = qa.get_effective_zero_point()
    else:
        zero_point = getattr(qa, "zero_point", None)
    if zero_point is None:
        return 0.0
    if torch.is_tensor(zero_point):
        if zero_point.numel() == 0:
            return 0.0
        return float(zero_point.detach().reshape(-1)[0].item())
    return float(zero_point)


def quantizer_uses_zero_point(qa):
    return bool(getattr(qa, "withzeropoint", False))


def quantizer_uses_zero_mean_laplace(qa):
    if quantizer_uses_zero_point(qa):
        return False
    return float(getattr(qa, "thd_neg", 0.0)) < 0.0 < float(getattr(qa, "thd_pos", 0.0))


def quantizer_uses_center_inflated_laplace(qa):
    if not bool(getattr(qa, "use_center_inflated_laplace", True)):
        return False
    return quantizer_uses_zero_mean_laplace(qa) or quantizer_uses_zero_point(qa)


def quantizer_center_symbol(qa):
    if quantizer_uses_zero_point(qa):
        return int(round(_get_quantizer_zero_point_value(qa)))
    return 0


def quantizer_symbol_domain_from_dequantized(x, qa):
    if hasattr(qa, "beta"):
        safe_scale = qa.s.clamp(min=1e-8)
        return (x - qa.beta) / safe_scale
    if hasattr(qa, "s"):
        return x / qa.s.clamp(min=1e-8)
    if quantizer_uses_zero_point(qa):
        return qa.get_effective_zero_point() + (x / qa.scale.clamp(min=1e-8)) if hasattr(qa, "get_effective_zero_point") else qa.zero_point + (x / qa.scale.clamp(min=1e-8))
    return x / qa.scale.clamp(min=1e-8)


def build_laplace_zero_mean_mask(qas, n_dims, n_blocks, device):
    flat_qas = list(qas)
    expected_qas = n_dims * n_blocks
    if len(flat_qas) != expected_qas:
        raise ValueError(f"Expected {expected_qas} quantizers, got {len(flat_qas)}")
    zero_mean_flags = [quantizer_uses_zero_mean_laplace(qa) for qa in flat_qas]
    return torch.tensor(zero_mean_flags, device=device, dtype=torch.bool).view(n_dims, n_blocks, 1)


def build_laplace_zero_inflation_mask(qas, n_dims, n_blocks, device):
    flat_qas = list(qas)
    expected_qas = n_dims * n_blocks
    if len(flat_qas) != expected_qas:
        raise ValueError(f"Expected {expected_qas} quantizers, got {len(flat_qas)}")
    zero_inflation_flags = [quantizer_uses_center_inflated_laplace(qa) for qa in flat_qas]
    return torch.tensor(zero_inflation_flags, device=device, dtype=torch.bool).view(n_dims, n_blocks, 1)


def build_laplace_center_symbol_tensor(qas, n_dims, n_blocks, device, dtype):
    flat_qas = list(qas)
    expected_qas = n_dims * n_blocks
    if len(flat_qas) != expected_qas:
        raise ValueError(f"Expected {expected_qas} quantizers, got {len(flat_qas)}")
    center_symbols = [quantizer_center_symbol(qa) for qa in flat_qas]
    return torch.tensor(center_symbols, device=device, dtype=dtype).view(n_dims, n_blocks, 1)


def _reshape_zero_mean_mask(zero_mean_mask, n_dims, n_blocks, device):
    if zero_mean_mask is None:
        return None
    zero_mean_mask = torch.as_tensor(zero_mean_mask, device=device, dtype=torch.bool)
    if zero_mean_mask.numel() != n_dims * n_blocks:
        raise ValueError(
            f"Expected {n_dims * n_blocks} zero-mean flags, got {int(zero_mean_mask.numel())}"
        )
    return zero_mean_mask.view(n_dims, n_blocks, 1)


def _reshape_zero_inflation_mask(zero_inflation_mask, n_dims, n_blocks, device):
    return _reshape_zero_mean_mask(zero_inflation_mask, n_dims, n_blocks, device)


def _reshape_center_symbol_tensor(center_symbol_values, n_dims, n_blocks, device, dtype):
    if center_symbol_values is None:
        return None
    center_symbol_values = torch.as_tensor(center_symbol_values, device=device, dtype=dtype)
    if center_symbol_values.numel() != n_dims * n_blocks:
        raise ValueError(
            f"Expected {n_dims * n_blocks} center-symbol values, got {int(center_symbol_values.numel())}"
        )
    return center_symbol_values.view(n_dims, n_blocks, 1)


def _estimate_laplace_block_params_from_packed(symbols, split_tensor, zero_mean_mask=None):
    block_lengths = split_tensor.to(dtype=symbols.dtype).view(1, -1, 1)
    max_block_len = symbols.shape[-1]
    valid_mask = (
        torch.arange(max_block_len, device=symbols.device, dtype=torch.long)
        .view(1, 1, -1)
        < split_tensor.view(1, -1, 1)
    )
    valid_mask_f = valid_mask.to(dtype=symbols.dtype)

    empirical_mean = (
        (symbols.detach() * valid_mask_f).sum(dim=-1)
        / block_lengths.squeeze(-1).clamp(min=1.0)
    ).unsqueeze(-1)
    if zero_mean_mask is None:
        block_mean = empirical_mean
    else:
        block_mean = torch.where(zero_mean_mask, torch.zeros_like(empirical_mean), empirical_mean)

    raw_scale = (
        ((symbols.detach() - block_mean).abs() * valid_mask_f).sum(dim=-1)
        / block_lengths.squeeze(-1).clamp(min=1.0)
    ).clamp(min=1e-6)
    return block_mean.squeeze(-1), raw_scale


def _estimate_zero_inflated_laplace_params_from_packed(
    symbols,
    split_tensor,
    zero_mean_mask=None,
    zero_inflation_mask=None,
    center_symbol_tensor=None,
):
    block_lengths = split_tensor.to(dtype=symbols.dtype).view(1, -1, 1)
    max_block_len = symbols.shape[-1]
    valid_mask = (
        torch.arange(max_block_len, device=symbols.device, dtype=torch.long)
        .view(1, 1, -1)
        < split_tensor.view(1, -1, 1)
    )
    valid_mask_f = valid_mask.to(dtype=symbols.dtype)

    empirical_mean = (
        (symbols.detach() * valid_mask_f).sum(dim=-1)
        / block_lengths.squeeze(-1).clamp(min=1.0)
    ).unsqueeze(-1)
    if zero_mean_mask is None:
        block_mean = empirical_mean
    else:
        block_mean = torch.where(zero_mean_mask, torch.zeros_like(empirical_mean), empirical_mean)
    if center_symbol_tensor is not None and zero_inflation_mask is not None:
        block_mean = torch.where(
            zero_inflation_mask,
            center_symbol_tensor.to(dtype=empirical_mean.dtype),
            block_mean,
        )

    full_scale = (
        ((symbols.detach() - block_mean).abs() * valid_mask_f).sum(dim=-1)
        / block_lengths.squeeze(-1).clamp(min=1.0)
    ).clamp(min=1e-6)

    if zero_inflation_mask is None:
        zero_probs = torch.zeros_like(full_scale)
        center_symbols = torch.zeros_like(block_mean.squeeze(-1))
        return block_mean.squeeze(-1), full_scale, zero_probs, center_symbols

    zero_inflation_mask = zero_inflation_mask.to(dtype=torch.bool)
    if center_symbol_tensor is None:
        center_symbol_tensor = torch.zeros_like(block_mean)
    center_symbols = center_symbol_tensor.squeeze(-1)
    center_matches = symbols.detach() == center_symbol_tensor.to(dtype=symbols.dtype)
    zero_counts = (center_matches & valid_mask).to(dtype=symbols.dtype).sum(dim=-1)
    smoothed_zero_probs = (zero_counts + 1.0) / (block_lengths.squeeze(-1) + 2.0)

    nonzero_mask = valid_mask & (~center_matches)
    nonzero_mask_f = nonzero_mask.to(dtype=symbols.dtype)
    nonzero_counts = nonzero_mask_f.sum(dim=-1)
    nonzero_scale = (
        ((symbols.detach() - block_mean).abs() * nonzero_mask_f).sum(dim=-1)
        / nonzero_counts.clamp(min=1.0)
    )
    nonzero_scale = torch.where(nonzero_counts > 0, nonzero_scale, torch.ones_like(nonzero_scale))

    zi_mask = zero_inflation_mask.squeeze(-1)
    raw_scale = torch.where(zi_mask, nonzero_scale, full_scale).clamp(min=1e-6)
    zero_probs = torch.where(zi_mask, smoothed_zero_probs, torch.zeros_like(smoothed_zero_probs))
    center_symbols = torch.where(zi_mask, center_symbols, torch.zeros_like(center_symbols))
    return block_mean.squeeze(-1), raw_scale, zero_probs, center_symbols


def _laplace_symbol_probs(quantized_symbols, block_scale, block_mean=None):
    safe_scale = block_scale.clamp(min=0.01)
    mean = None if block_mean is None else block_mean
    upper = _laplace_cdf(quantized_symbols + 0.5, safe_scale, mean=mean)
    lower = _laplace_cdf(quantized_symbols - 0.5, safe_scale, mean=mean)
    return (upper - lower).clamp(min=1e-9)


def _zero_inflated_laplace_symbol_probs(
    quantized_symbols,
    block_scale,
    block_zero_prob,
    block_mean=None,
    zero_inflation_mask=None,
    center_symbol_tensor=None,
):
    base_probs = _laplace_symbol_probs(quantized_symbols, block_scale, block_mean=block_mean)
    if zero_inflation_mask is None:
        return base_probs

    safe_zero_prob = block_zero_prob.clamp(min=0.0, max=1.0 - 1e-6)
    if center_symbol_tensor is None:
        center_symbol_tensor = torch.zeros_like(block_zero_prob)
    center_symbols = center_symbol_tensor.to(dtype=quantized_symbols.dtype).expand_as(quantized_symbols)
    center_bin_prob = _laplace_symbol_probs(center_symbols, block_scale, block_mean=block_mean)
    tail_norm = (1.0 - center_bin_prob).clamp(min=1e-9)
    nonzero_probs = ((1.0 - safe_zero_prob) * (base_probs / tail_norm)).clamp(min=1e-9)
    inflated_probs = torch.where(
        quantized_symbols == center_symbols,
        safe_zero_prob.clamp(min=1e-9),
        nonzero_probs,
    )
    return torch.where(zero_inflation_mask, inflated_probs, base_probs).clamp(min=1e-9)


def _estimate_laplace_bits(symbols, split_tensor, zero_mean_mask=None):
    max_block_len = symbols.shape[-1]
    valid_mask = (
        torch.arange(max_block_len, device=symbols.device, dtype=torch.long)
        .view(1, 1, -1)
        < split_tensor.view(1, -1, 1)
    )
    valid_mask_f = valid_mask.to(dtype=symbols.dtype)

    block_mean, raw_scale = _estimate_laplace_block_params_from_packed(
        symbols,
        split_tensor,
        zero_mean_mask=zero_mean_mask,
    )

    block_mean_expanded = block_mean.unsqueeze(-1)
    train_scale = raw_scale.clamp(min=0.01).unsqueeze(-1)  # [n_ch, n_blocks, 1]

    upper = _laplace_cdf(symbols + 0.5, train_scale, mean=block_mean_expanded)
    lower = _laplace_cdf(symbols - 0.5, train_scale, mean=block_mean_expanded)

    probs = (upper - lower).clamp(min=1e-9)
    per_element_bits = -torch.log2(probs)
    total_bits = (per_element_bits * valid_mask_f).sum()
    return total_bits, block_mean, raw_scale


def _estimate_zero_inflated_laplace_bits(
    symbols,
    split_tensor,
    zero_mean_mask=None,
    zero_inflation_mask=None,
    center_symbol_tensor=None,
):
    max_block_len = symbols.shape[-1]
    valid_mask = (
        torch.arange(max_block_len, device=symbols.device, dtype=torch.long)
        .view(1, 1, -1)
        < split_tensor.view(1, -1, 1)
    )
    valid_mask_f = valid_mask.to(dtype=symbols.dtype)

    block_mean, raw_scale, block_zero_probs, block_center_symbols = _estimate_zero_inflated_laplace_params_from_packed(
        symbols,
        split_tensor,
        zero_mean_mask=zero_mean_mask,
        zero_inflation_mask=zero_inflation_mask,
        center_symbol_tensor=center_symbol_tensor,
    )

    block_mean_expanded = block_mean.unsqueeze(-1)
    train_scale = raw_scale.clamp(min=0.01).unsqueeze(-1)
    train_zero_probs = block_zero_probs.unsqueeze(-1)
    probs = _zero_inflated_laplace_symbol_probs(
        symbols,
        train_scale,
        train_zero_probs,
        block_mean=block_mean_expanded,
        zero_inflation_mask=zero_inflation_mask,
        center_symbol_tensor=block_center_symbols.unsqueeze(-1),
    )
    per_element_bits = -torch.log2(probs)
    total_bits = (per_element_bits * valid_mask_f).sum()
    return total_bits, block_mean, raw_scale, block_zero_probs, block_center_symbols


def _laplace_bits_with_tail_clip(quantized_symbols, block_scale, block_mean=None, sigma=3.0):
    """Compute -log2(prob) per symbol using input-clamp + probability-floor.

    Numerics follow the same three-layer protection as
    ``_estimate_laplace_bits``:

    1. ``block_scale`` lower-bounded at 0.01 (caller's responsibility, but
       also re-applied here for safety) → prevents an extremely peaked
       distribution where the CDF difference collapses.
    2. Input 3-sigma clamp *before* the CDF: ``x ← clamp(x, -σb, σb)``.
       Outlier symbols are pulled to the boundary so their CDF interval
       probability remains a reasonable positive number and the gradient
       still flows through the symbol value.（没做）
    3. ``probs.clamp(min=1e-5)`` → max penalty ≈ 16.6 bits/element,
       prevents log(0) and keeps gradient magnitudes bounded.

    Parameters
    ----------
    quantized_symbols : Tensor
        Broadcastable with ``block_scale``.
    block_scale : Tensor
        Laplace scale b, broadcast-compatible with ``quantized_symbols``.
    sigma : float
        Truncation threshold multiplier. Default 3.

    Returns
    -------
    per_element_bits : Tensor, same shape as ``quantized_symbols``.
    """
    # 【保护1】scale 下界 0.01
    safe_scale = block_scale.clamp(min=0.01)

    mean = None if block_mean is None else block_mean
    upper = _laplace_cdf(quantized_symbols + 0.5, safe_scale, mean=mean)
    lower = _laplace_cdf(quantized_symbols - 0.5, safe_scale, mean=mean)

    # 【保护2】likelihood 下界 1e-9 (最大罚 ~29.9 bits)
    probs = (upper - lower).clamp(min=1e-9)
    return -torch.log2(probs)


def _zero_inflated_laplace_bits_with_tail_clip(
    quantized_symbols,
    block_scale,
    block_zero_prob,
    block_mean=None,
    zero_inflation_mask=None,
    center_symbol_tensor=None,
    sigma=3.0,
):
    del sigma
    probs = _zero_inflated_laplace_symbol_probs(
        quantized_symbols,
        block_scale.clamp(min=0.01),
        block_zero_prob,
        block_mean=block_mean,
        zero_inflation_mask=zero_inflation_mask,
        center_symbol_tensor=center_symbol_tensor,
    )
    return -torch.log2(probs.clamp(min=1e-9))


def estimate_laplace_block_params(symbols, split, qas=None, zero_mean_flags=None):
    squeeze_output = False
    if symbols.dim() == 1:
        symbols = symbols.reshape(-1, 1)
        squeeze_output = True
    if symbols.dim() != 2:
        raise ValueError(f"Expected symbols to be 1D or 2D, got shape {tuple(symbols.shape)}")

    split_tensor, block_ids, local_ids = _build_block_index_tensors(split, symbols.device)
    max_block_len = int(split_tensor.max().item())
    xt = symbols.transpose(0, 1).contiguous()
    packed = xt.new_zeros((symbols.shape[1], len(split), max_block_len))
    packed[:, block_ids, local_ids] = xt

    zero_mean_mask = None
    if qas is not None:
        zero_mean_mask = build_laplace_zero_mean_mask(
            qas,
            n_dims=symbols.shape[1],
            n_blocks=len(split),
            device=symbols.device,
        )
    elif zero_mean_flags is not None:
        zero_mean_mask = _reshape_zero_mean_mask(
            zero_mean_flags,
            n_dims=symbols.shape[1],
            n_blocks=len(split),
            device=symbols.device,
        )

    block_means, block_scales = _estimate_laplace_block_params_from_packed(
        packed,
        split_tensor,
        zero_mean_mask=zero_mean_mask,
    )

    if squeeze_output:
        return block_means.reshape(-1), block_scales.reshape(-1)
    return block_means, block_scales


def estimate_zero_inflated_laplace_block_params(
    symbols,
    split,
    qas=None,
    zero_mean_flags=None,
    zero_inflation_flags=None,
    center_symbol_values=None,
):
    squeeze_output = False
    if symbols.dim() == 1:
        symbols = symbols.reshape(-1, 1)
        squeeze_output = True
    if symbols.dim() != 2:
        raise ValueError(f"Expected symbols to be 1D or 2D, got shape {tuple(symbols.shape)}")

    split_tensor, block_ids, local_ids = _build_block_index_tensors(split, symbols.device)
    max_block_len = int(split_tensor.max().item())
    xt = symbols.transpose(0, 1).contiguous()
    packed = xt.new_zeros((symbols.shape[1], len(split), max_block_len))
    packed[:, block_ids, local_ids] = xt

    zero_mean_mask = None
    zero_inflation_mask = None
    center_symbol_tensor = None
    if qas is not None:
        zero_mean_mask = build_laplace_zero_mean_mask(
            qas,
            n_dims=symbols.shape[1],
            n_blocks=len(split),
            device=symbols.device,
        )
        zero_inflation_mask = build_laplace_zero_inflation_mask(
            qas,
            n_dims=symbols.shape[1],
            n_blocks=len(split),
            device=symbols.device,
        )
        center_symbol_tensor = build_laplace_center_symbol_tensor(
            qas,
            n_dims=symbols.shape[1],
            n_blocks=len(split),
            device=symbols.device,
            dtype=symbols.dtype,
        )
    elif zero_mean_flags is not None:
        zero_mean_mask = _reshape_zero_mean_mask(
            zero_mean_flags,
            n_dims=symbols.shape[1],
            n_blocks=len(split),
            device=symbols.device,
        )
        inferred_flags = zero_mean_flags if zero_inflation_flags is None else zero_inflation_flags
        zero_inflation_mask = _reshape_zero_inflation_mask(
            inferred_flags,
            n_dims=symbols.shape[1],
            n_blocks=len(split),
            device=symbols.device,
        )
        center_symbol_tensor = _reshape_center_symbol_tensor(
            center_symbol_values,
            n_dims=symbols.shape[1],
            n_blocks=len(split),
            device=symbols.device,
            dtype=symbols.dtype,
        )
    elif zero_inflation_flags is not None:
        zero_inflation_mask = _reshape_zero_inflation_mask(
            zero_inflation_flags,
            n_dims=symbols.shape[1],
            n_blocks=len(split),
            device=symbols.device,
        )
        center_symbol_tensor = _reshape_center_symbol_tensor(
            center_symbol_values,
            n_dims=symbols.shape[1],
            n_blocks=len(split),
            device=symbols.device,
            dtype=symbols.dtype,
        )

    block_means, block_scales, block_zero_probs, block_center_symbols = _estimate_zero_inflated_laplace_params_from_packed(
        packed,
        split_tensor,
        zero_mean_mask=zero_mean_mask,
        zero_inflation_mask=zero_inflation_mask,
        center_symbol_tensor=center_symbol_tensor,
    )

    if squeeze_output:
        return (
            block_means.reshape(-1),
            block_scales.reshape(-1),
            block_zero_probs.reshape(-1),
            block_center_symbols.reshape(-1),
        )
    return block_means, block_scales, block_zero_probs, block_center_symbols


def estimate_zero_mean_laplace_block_scales(symbols, split):
    zero_mean_flags = torch.ones((symbols.shape[1] if symbols.dim() == 2 else 1, len(split)), dtype=torch.bool)
    _, block_scales = estimate_laplace_block_params(
        symbols,
        split,
        zero_mean_flags=zero_mean_flags,
    )
    return block_scales


def _sync_profile_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _flatten_quant_buffer(buffer, default_value, device, dtype):
    if buffer is None or buffer.nelement() == 0:
        return torch.as_tensor([default_value], device=device, dtype=dtype)
    return buffer.detach().reshape(1).to(device=device, dtype=dtype)


def _write_quant_buffer(module, name, value):
    target = getattr(module, name)
    value = value.detach().to(device=target.device, dtype=target.dtype)
    if target.nelement() == value.nelement():
        target.data.copy_(value.reshape_as(target))
    else:
        setattr(module, name, value.clone())


def batched_quantize_blocks(
    x,
    split,
    qas,
    return_symbols=False,
    return_trans=False,
    return_ans_bits=False,
    return_profile=False,
    profile_time=False,
    step_override=None,
    lock_quant_params=False,
    override_uses_vanilla_zero_point=False,
):
    """
    Quantize all attribute/block pairs in a single batched tensor path.
    x can be [N] or [N, D], while qas are arranged in dim-major block order.
    """
    squeeze_output = False
    if x.dim() == 1:
        x = x.reshape(-1, 1)
        squeeze_output = True
    elif x.dim() != 2:
        raise ValueError(f"Expected x to be 1D or 2D, got shape {tuple(x.shape)}")

    quant_profile = {
        "index_prepare": 0.0,
        "pack_scatter": 0.0,
        "block_stat": 0.0,
        "quant_param_update": 0.0,
        "quant_param_stack": 0.0,
        "quant_core": 0.0,
        "entropy_bits": 0.0,
        "trans_collect": 0.0,
        "total": 0.0,
    } if return_profile else None
    total_start = None
    if quant_profile is not None and profile_time:
        _sync_profile_device(x.device)
        total_start = time.perf_counter()

    flat_qas = list(qas)
    n_points, n_dims = x.shape
    n_blocks = len(split)
    expected_qas = n_dims * n_blocks
    if len(flat_qas) != expected_qas:
        raise ValueError(f"Expected {expected_qas} quantizers, got {len(flat_qas)}")

    stage_start = None
    if quant_profile is not None and profile_time:
        _sync_profile_device(x.device)
        stage_start = time.perf_counter()
    split_tensor, block_ids, local_ids = _build_block_index_tensors(split, x.device)
    if int(split_tensor.sum().item()) != n_points:
        raise ValueError(f"Split sum {int(split_tensor.sum().item())} does not match input length {n_points}")
    if quant_profile is not None and profile_time:
        _sync_profile_device(x.device)
        quant_profile["index_prepare"] = time.perf_counter() - stage_start

    if quant_profile is not None and profile_time:
        _sync_profile_device(x.device)
        stage_start = time.perf_counter()
    max_block_len = int(split_tensor.max().item())
    xt = x.transpose(0, 1).contiguous()
    packed = xt.new_zeros((n_dims, n_blocks, max_block_len))
    packed[:, block_ids, local_ids] = xt

    valid_mask = (
        torch.arange(max_block_len, device=x.device, dtype=torch.long)
        .view(1, 1, -1)
        < split_tensor.view(1, n_blocks, 1)
    )
    valid_mask_f = valid_mask.to(dtype=x.dtype)
    abs_packed = packed.abs()
    block_lengths = split_tensor.to(dtype=x.dtype).view(1, n_blocks)
    if quant_profile is not None and profile_time:
        _sync_profile_device(x.device)
        quant_profile["pack_scatter"] = time.perf_counter() - stage_start

    first_qa = flat_qas[0]
    use_step_override = step_override is not None
    use_beta_param = hasattr(first_qa, "beta")
    override_offsets = None
    if use_step_override:
        override_steps = torch.as_tensor(step_override, device=x.device, dtype=x.dtype)
        if override_steps.dim() == 2:
            override_steps = override_steps.unsqueeze(-1)
        elif override_steps.dim() != 3:
            raise ValueError(
                f"Expected step_override to have shape [n_dims, n_blocks] or [n_dims, n_blocks, 1], got {tuple(override_steps.shape)}"
            )
        if override_steps.shape[0] != n_dims or override_steps.shape[1] != n_blocks:
            raise ValueError(
                f"step_override shape {tuple(override_steps.shape)} does not match expected ({n_dims}, {n_blocks}, 1)"
            )
        override_steps = override_steps.clamp(min=1e-8)
        if use_beta_param:
            beta_entries = []
            for qa in flat_qas:
                beta_entries.append(qa.beta.reshape(1).to(device=x.device, dtype=x.dtype))
            override_offsets = torch.stack(beta_entries, dim=0).view(n_dims, n_blocks, 1)
        else:
            override_offsets = override_steps.new_zeros((n_dims, n_blocks, 1))
        if hasattr(first_qa, "scale") and bool(override_uses_vanilla_zero_point):
            if quant_profile is not None and profile_time:
                _sync_profile_device(x.device)
                stage_start = time.perf_counter()
            neg_fill = torch.full_like(packed, torch.finfo(packed.dtype).min)
            pos_fill = torch.full_like(packed, torch.finfo(packed.dtype).max)
            block_max = torch.where(valid_mask, packed, neg_fill).amax(dim=-1)
            block_min = torch.where(valid_mask, packed, pos_fill).amin(dim=-1)
            flat_block_max = block_max.reshape(-1).detach()
            flat_block_min = block_min.reshape(-1).detach()
            flat_override_steps = override_steps.reshape(-1).detach()
            with_zero_mask = torch.tensor(
                [bool(getattr(qa, "withzeropoint", False)) for qa in flat_qas],
                device=x.device,
                dtype=torch.bool,
            )
            all_positive_mask = torch.tensor(
                [bool(getattr(qa, "all_positive", False)) for qa in flat_qas],
                device=x.device,
                dtype=torch.bool,
            )
            thd_neg_flat = torch.tensor(
                [float(qa.thd_neg) for qa in flat_qas],
                device=x.device,
                dtype=x.dtype,
            )
            thd_pos_flat = torch.tensor(
                [float(qa.thd_pos) for qa in flat_qas],
                device=x.device,
                dtype=x.dtype,
            )
            prev_max_flat = torch.cat(
                [
                    _flatten_quant_buffer(qa.max_val, float("-inf"), x.device, x.dtype)
                    for qa in flat_qas
                ],
                dim=0,
            )
            prev_min_flat = torch.cat(
                [
                    _flatten_quant_buffer(qa.min_val, float("inf"), x.device, x.dtype)
                    for qa in flat_qas
                ],
                dim=0,
            )
            updated_max_flat = prev_max_flat.clone()
            updated_min_flat = prev_min_flat.clone()
            updated_zero_flat = torch.zeros_like(prev_max_flat)
            if with_zero_mask.any():
                zp_max = torch.maximum(prev_max_flat[with_zero_mask], flat_block_max[with_zero_mask])
                zp_min = torch.minimum(prev_min_flat[with_zero_mask], flat_block_min[with_zero_mask])
                zp_all_positive = all_positive_mask[with_zero_mask]
                if zp_all_positive.any():
                    zp_max = torch.where(zp_all_positive, torch.clamp(zp_max, min=0.0), zp_max)
                    zp_min = torch.where(zp_all_positive, torch.clamp(zp_min, max=0.0), zp_min)
                zp_max = torch.maximum(zp_max, zp_min + 1e-8)
                zp_scale = flat_override_steps[with_zero_mask].clamp(min=1e-8)
                zp_zero = thd_pos_flat[with_zero_mask] - (zp_max / zp_scale)
                zp_zero = torch.clamp(
                    zp_zero,
                    min=thd_neg_flat[with_zero_mask],
                    max=thd_pos_flat[with_zero_mask],
                ).round()
                updated_max_flat[with_zero_mask] = zp_max
                updated_min_flat[with_zero_mask] = zp_min
                updated_zero_flat[with_zero_mask] = zp_zero
            with torch.no_grad():
                for idx, qa in enumerate(flat_qas):
                    _write_quant_buffer(qa, "scale", flat_override_steps[idx])
                    if hasattr(qa, "max_val"):
                        _write_quant_buffer(qa, "max_val", updated_max_flat[idx])
                    if hasattr(qa, "min_val"):
                        _write_quant_buffer(qa, "min_val", updated_min_flat[idx])
                    if hasattr(qa, "zero_point"):
                        qa.zero_point.data.copy_(
                            updated_zero_flat[idx]
                            .to(device=qa.zero_point.device, dtype=qa.zero_point.dtype)
                            .reshape_as(qa.zero_point)
                        )
            override_offsets = updated_zero_flat.view(n_dims, n_blocks, 1)
            if quant_profile is not None and profile_time:
                _sync_profile_device(x.device)
                quant_profile["quant_param_update"] = time.perf_counter() - stage_start
    elif hasattr(first_qa, "init_yet"):
        if quant_profile is not None and profile_time:
            _sync_profile_device(x.device)
            stage_start = time.perf_counter()
        block_abs_mean = (abs_packed * valid_mask_f).sum(dim=-1) / block_lengths
        if quant_profile is not None and profile_time:
            _sync_profile_device(x.device)
            quant_profile["block_stat"] = time.perf_counter() - stage_start
            _sync_profile_device(x.device)
            stage_start = time.perf_counter()
        for dim_idx in range(n_dims):
            base = dim_idx * n_blocks
            for block_idx in range(n_blocks):
                qa = flat_qas[base + block_idx]
                if not qa.init_yet:
                    block_len = int(split[block_idx])
                    block_values = packed[dim_idx, block_idx, :block_len]
                    if hasattr(qa, "init_from"):
                        qa.init_from(block_values)
                    else:
                        init_scale = block_abs_mean[dim_idx, block_idx] * 2 / (float(qa.thd_pos) ** 0.5)
                        with torch.no_grad():
                            qa.s.data.fill_(init_scale.item())
                        qa.init_yet = True
        if quant_profile is not None and profile_time:
            quant_profile["quant_param_update"] = time.perf_counter() - stage_start
    elif hasattr(first_qa, "scale"):
        if quant_profile is not None and profile_time:
            _sync_profile_device(x.device)
            stage_start = time.perf_counter()
        neg_fill = torch.full_like(packed, torch.finfo(packed.dtype).min)
        pos_fill = torch.full_like(packed, torch.finfo(packed.dtype).max)
        block_abs_max = torch.where(valid_mask, abs_packed, abs_packed.new_zeros(1)).amax(dim=-1)
        block_max = torch.where(valid_mask, packed, neg_fill).amax(dim=-1)
        block_min = torch.where(valid_mask, packed, pos_fill).amin(dim=-1)
        if quant_profile is not None and profile_time:
            _sync_profile_device(x.device)
            quant_profile["block_stat"] = time.perf_counter() - stage_start
            _sync_profile_device(x.device)
            stage_start = time.perf_counter()
        flat_block_max = block_max.reshape(-1).detach()
        flat_block_min = block_min.reshape(-1).detach()
        flat_block_abs_max = block_abs_max.reshape(-1).detach()

        with_zero_mask = torch.tensor(
            [bool(getattr(qa, "withzeropoint", False)) for qa in flat_qas],
            device=x.device,
            dtype=torch.bool,
        )
        all_positive_mask = torch.tensor(
            [bool(getattr(qa, "all_positive", False)) for qa in flat_qas],
            device=x.device,
            dtype=torch.bool,
        )
        thd_neg_flat = torch.tensor(
            [float(qa.thd_neg) for qa in flat_qas],
            device=x.device,
            dtype=x.dtype,
        )
        thd_pos_flat = torch.tensor(
            [float(qa.thd_pos) for qa in flat_qas],
            device=x.device,
            dtype=x.dtype,
        )
        qmax_safe_flat = torch.clamp(thd_pos_flat, min=1.0)
        prev_max_flat = torch.cat(
            [
                _flatten_quant_buffer(qa.max_val, float("-inf"), x.device, x.dtype)
                for qa in flat_qas
            ],
            dim=0,
        )
        prev_min_flat = torch.cat(
            [
                _flatten_quant_buffer(qa.min_val, float("inf"), x.device, x.dtype)
                for qa in flat_qas
            ],
            dim=0,
        )

        updated_max_flat = prev_max_flat.clone()
        updated_min_flat = prev_min_flat.clone()
        updated_scale_flat = torch.ones_like(prev_max_flat)
        updated_zero_flat = torch.zeros_like(prev_max_flat)

        if with_zero_mask.any():
            zp_max = torch.maximum(prev_max_flat[with_zero_mask], flat_block_max[with_zero_mask])
            zp_min = torch.minimum(prev_min_flat[with_zero_mask], flat_block_min[with_zero_mask])
            zp_all_positive = all_positive_mask[with_zero_mask]
            if zp_all_positive.any():
                zp_max = torch.where(zp_all_positive, torch.clamp(zp_max, min=0.0), zp_max)
                zp_min = torch.where(zp_all_positive, torch.clamp(zp_min, max=0.0), zp_min)
            zp_max = torch.maximum(zp_max, zp_min + 1e-8)
            qrange = torch.clamp(
                thd_pos_flat[with_zero_mask] - thd_neg_flat[with_zero_mask],
                min=1.0,
            )
            zp_scale = torch.clamp((zp_max - zp_min) / qrange, min=1e-8)
            zp_zero = thd_pos_flat[with_zero_mask] - (zp_max / zp_scale)
            zp_zero = torch.clamp(
                zp_zero,
                min=thd_neg_flat[with_zero_mask],
                max=thd_pos_flat[with_zero_mask],
            ).round()
            updated_max_flat[with_zero_mask] = zp_max
            updated_min_flat[with_zero_mask] = zp_min
            updated_scale_flat[with_zero_mask] = zp_scale
            updated_zero_flat[with_zero_mask] = zp_zero

        positive_mask = (~with_zero_mask) & all_positive_mask
        if positive_mask.any():
            pos_max = torch.maximum(prev_max_flat[positive_mask], flat_block_max[positive_mask])
            pos_max = torch.clamp(pos_max, min=1e-8)
            updated_max_flat[positive_mask] = pos_max
            updated_min_flat[positive_mask] = 0.0
            updated_scale_flat[positive_mask] = torch.clamp(
                pos_max / qmax_safe_flat[positive_mask],
                min=1e-8,
            )
            updated_zero_flat[positive_mask] = 0.0

        signed_mask = (~with_zero_mask) & (~all_positive_mask)
        if signed_mask.any():
            signed_abs_max = torch.maximum(
                prev_max_flat[signed_mask],
                flat_block_abs_max[signed_mask],
            )
            signed_abs_max = torch.clamp(signed_abs_max, min=1e-8)
            updated_max_flat[signed_mask] = signed_abs_max
            updated_min_flat[signed_mask] = -signed_abs_max
            updated_scale_flat[signed_mask] = torch.clamp(
                signed_abs_max / qmax_safe_flat[signed_mask],
                min=1e-8,
            )
            updated_zero_flat[signed_mask] = 0.0

        with torch.no_grad():
            for idx, qa in enumerate(flat_qas):
                _write_quant_buffer(qa, "max_val", updated_max_flat[idx])
                _write_quant_buffer(qa, "min_val", updated_min_flat[idx])
                _write_quant_buffer(qa, "scale", updated_scale_flat[idx])
                qa.zero_point.data.copy_(
                    updated_zero_flat[idx]
                    .to(device=qa.zero_point.device, dtype=qa.zero_point.dtype)
                    .reshape_as(qa.zero_point)
                )
        if quant_profile is not None and profile_time:
            quant_profile["quant_param_update"] = time.perf_counter() - stage_start
    else:
        raise ValueError(f"Unsupported quantizer type: {type(first_qa)}")

    if quant_profile is not None and profile_time:
        _sync_profile_device(x.device)
        stage_start = time.perf_counter()
    neg_entries = []
    pos_entries = []
    if use_step_override:
        scales = override_steps
        offsets = override_offsets
        for qa in flat_qas:
            neg_entries.append(scales.new_tensor(float(qa.thd_neg)))
            pos_entries.append(scales.new_tensor(float(qa.thd_pos)))
    else:
        scale_entries = []
        offset_entries = []
        for qa in flat_qas:
            if hasattr(qa, "s"):
                scale_entries.append(qa.s.reshape(1))
            else:
                scale_entries.append(qa.scale.reshape(1))
            if hasattr(qa, "beta"):
                offset_entries.append(qa.beta.reshape(1).to(dtype=scale_entries[-1].dtype))
            elif hasattr(qa, "zero_point"):
                offset_entries.append(qa.zero_point.reshape(1).to(dtype=scale_entries[-1].dtype))
            else:
                offset_entries.append(scale_entries[-1].new_zeros(1))
            neg_entries.append(scale_entries[-1].new_tensor(float(qa.thd_neg)))
            pos_entries.append(scale_entries[-1].new_tensor(float(qa.thd_pos)))
        scales = torch.stack(scale_entries, dim=0).view(n_dims, n_blocks, 1)
        offsets = torch.stack(offset_entries, dim=0).view(n_dims, n_blocks, 1)
    thd_neg = torch.stack(neg_entries, dim=0).view(n_dims, n_blocks, 1)
    thd_pos = torch.stack(pos_entries, dim=0).view(n_dims, n_blocks, 1)
    if quant_profile is not None and profile_time:
        _sync_profile_device(x.device)
        quant_profile["quant_param_stack"] = time.perf_counter() - stage_start

    if quant_profile is not None and profile_time:
        _sync_profile_device(x.device)
        stage_start = time.perf_counter()
    if hasattr(first_qa, "s") and not use_step_override:
        grad_factors = 1.0 / torch.sqrt(thd_pos.clamp(min=1.0) * block_lengths.view(1, n_blocks, 1))
        scales = grad_scale(scales, grad_factors)
        if use_beta_param or getattr(first_qa, "withzeropoint", False):
            offsets = grad_scale(offsets, grad_factors)

    if use_beta_param:
        x_norm = (packed - offsets) / scales
    elif getattr(first_qa, "withzeropoint", False):
        x_norm = offsets + (packed / scales)
    else:
        x_norm = packed / scales
    x_clamped = torch.clamp(x_norm, thd_neg, thd_pos)
    x_q = round_pass(x_clamped)
    if use_beta_param:
        dequantized = x_q * scales + offsets
    elif getattr(first_qa, "withzeropoint", False):
        dequantized = (x_q - offsets) * scales
    else:
        dequantized = x_q * scales

    flat_clamped = x_clamped[:, block_ids, local_ids]
    flat_symbols = x_q[:, block_ids, local_ids].transpose(0, 1).contiguous()
    flat_dequantized = dequantized[:, block_ids, local_ids].transpose(0, 1).contiguous()
    if quant_profile is not None and profile_time:
        _sync_profile_device(x.device)
        quant_profile["quant_core"] = time.perf_counter() - stage_start

    ans_bits = x.new_zeros(())
    entropy_scales = None
    encode_mode = getattr(first_qa, "encode", "deflate").lower()
    needs_clamped_cache = any(getattr(qa, "encode", "deflate").lower() == "ans" for qa in flat_qas)
    if quant_profile is not None and profile_time:
        _sync_profile_device(x.device)
        stage_start = time.perf_counter()
    if return_ans_bits and encode_mode == "laplace":
        zero_mean_mask = build_laplace_zero_mean_mask(
            flat_qas,
            n_dims=n_dims,
            n_blocks=n_blocks,
            device=x_q.device,
        )
        zero_inflation_mask = build_laplace_zero_inflation_mask(
            flat_qas,
            n_dims=n_dims,
            n_blocks=n_blocks,
            device=x_q.device,
        )
        center_symbol_tensor = build_laplace_center_symbol_tensor(
            flat_qas,
            n_dims=n_dims,
            n_blocks=n_blocks,
            device=x_q.device,
            dtype=x_q.dtype,
        )
        ans_bits, entropy_means, entropy_scales, _, _ = _estimate_zero_inflated_laplace_bits(
            x_q,
            split_tensor,
            zero_mean_mask=zero_mean_mask,
            zero_inflation_mask=zero_inflation_mask,
            center_symbol_tensor=center_symbol_tensor,
        )
    if return_ans_bits or needs_clamped_cache:
        for dim_idx in range(n_dims):
            base = dim_idx * n_blocks
            start = 0
            for block_idx, length in enumerate(split):
                qa = flat_qas[base + block_idx]
                if getattr(qa, "encode", "deflate").lower() == "ans":
                    clamped_block = flat_clamped[dim_idx, start:start + length]
                    if return_ans_bits and qa.entropy_bottleneck is not None and length > 0:
                        _, likelihoods = qa.entropy_bottleneck(clamped_block.view(1, 1, -1, 1))
                        # Use a higher clamp floor (1e-6) and cap per-element bits
                        # to prevent extreme gradients during backprop.
                        per_elem_bits = torch.clamp(-torch.log2(likelihoods.clamp(min=1e-6)), max=32.0)
                        ans_bits = ans_bits + per_elem_bits.sum()
                    qa.last_clamped = None
                start += length
    if quant_profile is not None and profile_time:
        _sync_profile_device(x.device)
        quant_profile["entropy_bits"] = time.perf_counter() - stage_start

    trans = None
    if return_trans:
        if quant_profile is not None and profile_time:
            _sync_profile_device(x.device)
            stage_start = time.perf_counter()
        trans = []
        if use_step_override:
            flat_scales = scales.reshape(-1)
            if use_beta_param:
                flat_zeros = -(offsets.reshape(-1) / flat_scales.clamp(min=1e-8))
            else:
                flat_zeros = offsets.reshape(-1)
            for idx in range(flat_scales.numel()):
                trans.extend([flat_scales[idx].item(), flat_zeros[idx].item()])
        else:
            for qa in flat_qas:
                i_scale, i_zp, _ = qa.get_quant_params()
                trans.extend([i_scale.item(), i_zp.item()])
        if quant_profile is not None and profile_time:
            quant_profile["trans_collect"] = time.perf_counter() - stage_start

    if squeeze_output:
        flat_dequantized = flat_dequantized.reshape(-1)
        flat_symbols = flat_symbols.reshape(-1)

    if quant_profile is not None and profile_time:
        _sync_profile_device(x.device)
        quant_profile["total"] = time.perf_counter() - total_start

    if return_symbols and return_trans and return_ans_bits:
        result = (flat_dequantized, flat_symbols, trans, ans_bits)
    elif return_symbols and return_trans:
        result = (flat_dequantized, flat_symbols, trans)
    elif return_symbols and return_ans_bits:
        result = (flat_dequantized, flat_symbols, ans_bits)
    elif return_symbols:
        result = (flat_dequantized, flat_symbols)
    elif return_trans and return_ans_bits:
        result = (flat_dequantized, trans, ans_bits)
    elif return_trans:
        result = (flat_dequantized, trans)
    elif return_ans_bits:
        result = (flat_dequantized, ans_bits)
    else:
        result = flat_dequantized

    if return_profile:
        if isinstance(result, tuple):
            return (*result, quant_profile)
        return result, quant_profile
    return result

class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
class ALSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta):
        # assert alpha > 0, "alpha={}".format(alpha)
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp
        w_q = Round.apply(torch.div((weight - beta), alpha).clamp(Qn, Qp))
        w_q = w_q * alpha + beta
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = (weight - beta) / alpha
        smaller = (q_w < Qn).float() #bool值转浮点值，1.0或者0.0
        bigger = (q_w > Qp).float() #bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller -bigger #得到位于量化区间的index
        grad_alpha = ((smaller * Qn + bigger * Qp + 
            between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        #在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight
        #返回的梯度要和forward的参数对应起来
        return grad_weight, grad_alpha,  None, None, None, grad_beta


class LSQPlusActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False,batch_init = 20):
        #activations 没有per-channel这个选项的
        super(LSQPlusActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** self.a_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (self.a_bits - 1)
            self.Qp = 2 ** (self.a_bits - 1) - 1
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        # self.beta = torch.nn.Parameter(torch.tensor([float(0)]))
        self.beta = torch.nn.Parameter(torch.tensor([float(-1e-9)]), requires_grad=True)
        self.init_state = 0

    def forward(self, activation):
        #V1
        # print(self.a_bits, self.batch_init)
        if self.a_bits == 32:
            q_a = activation
        elif self.a_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits != 1
        else:
            if self.init_state==0:
                self.g = 1.0/math.sqrt(activation.numel() * self.Qp)
                self.init_state += 1
            q_a = ALSQPlus.apply(activation, self.s, self.g, self.Qn, self.Qp, self.beta)
            # print(self.s, self.beta)
        return q_a

def grad_scale(x, scale):
    y = x
    y_grad = x * scale 
    return (y - y_grad).detach() + y_grad 

def round_pass(x):
    y = x.round()
    y_grad = x 
    return (y - y_grad).detach() + y_grad

class Quantizer(nn.Module):
    def __init__(self, bit):
        super().__init__()
        self.bit = bit  # 保存 bit 属性
        self.use_center_inflated_laplace = True

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError

    def get_quant_params(self):
        raise NotImplementedError


class IdentityQuan(Quantizer):
    def __init__(self, bit=None, *args, **kwargs):
        super().__init__(bit)
        assert bit is None, 'The bit-width of identity quantizer must be None'

    def forward(self, x):
        return x


class LsqQuan(Quantizer):
    def __init__(self, bit, init_yet, all_positive=True, symmetric=False, encode="deflate", channels=1, shared_eb=None):
        super().__init__(bit)
        
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.s = nn.Parameter(torch.ones(1))
        self.init_yet = init_yet
        
        # 🌟 新增：编码器选择与熵模型初始化
        self.encode = encode
        if self.encode.lower() == "ans":
            self.entropy_bottleneck = shared_eb
        else:
            self.entropy_bottleneck = None
        self.last_likelihoods = None
        self.last_clamped = None

    def init_from(self, x, *args, **kwargs):
        with torch.no_grad():
            # LSQ论文里的初始化方法，但是这个初始化对于我的数据来说好像不太行
            #self.s.data.fill_(x.detach().abs().mean().item() * 2 / (self.thd_pos ** 0.5))
            # 以下为min-max初始化
            min_val = x.detach().min()
            max_val = x.detach().max()
            if max_val == min_val:
                max_val = max_val + 1e-5
            scale_init = (max_val - min_val) / (self.thd_pos - self.thd_neg)
            self.s.data.fill_(scale_init.item())
        self.init_yet = True
        # print('quant_utils.py Line 62:', self.s)  # 打印初始化后的s
    
    def forward(self, x):
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        # 1. 归一化与截断 (LSQ 原本逻辑)
        x_norm = x / s_scale
        x_clamped = torch.clamp(x_norm, self.thd_neg, self.thd_pos)
        
        # 2. 🌟 新增：缓存 clamped 值供外部批处理，不再此处直接调用 EB 以节省开销
        if self.encode.lower() == "ans":
            # _, likelihoods = self.entropy_bottleneck(x_clamped.view(1, 1, -1, 1))
            # self.last_likelihoods = likelihoods.view(x.shape)
            self.last_clamped = x_clamped
        
        # 3. 直通估计器 (STE) 离散化与反量化
        x_q = round_pass(x_clamped)
        x_q = x_q * s_scale
        return x_q

    def get_quant_params(self):
        return self.s, torch.tensor(0.0, device=self.s.device), self.thd_neg < 0


class LsqPlusQuan(Quantizer):
    def __init__(
        self,
        bit,
        init_yet,
        all_positive=True,
        symmetric=False,
        encode="deflate",
        channels=1,
        shared_eb=None,
        learn_beta=False,
    ):
        super().__init__(bit)

        self.all_positive = all_positive
        self.symmetric = symmetric
        self.learn_beta = bool(learn_beta)
        self.offset_mode = "beta" if self.learn_beta else "zero_point"

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.s = nn.Parameter(torch.ones(1))
        self.withzeropoint = True
        if self.learn_beta:
            self.beta = nn.Parameter(torch.zeros(1))
        else:
            self.zero_point = nn.Parameter(torch.zeros(1))
        self.init_yet = init_yet

        self.encode = encode
        if self.encode.lower() == "ans":
            self.entropy_bottleneck = shared_eb
        else:
            self.entropy_bottleneck = None
        self.last_likelihoods = None
        self.last_clamped = None

    def init_from(self, x, *args, **kwargs):
        with torch.no_grad():
            min_val = x.detach().min()
            max_val = x.detach().max()
            scale, zero_point = calcScaleZeroPoint(
                min_val,
                max_val,
                num_bits=self.bit,
                qmin=float(self.thd_neg),
                qmax=float(self.thd_pos),
            )
            scale = torch.clamp(scale.reshape_as(self.s), min=1e-8)
            self.s.data.copy_(scale)
            if self.learn_beta:
                beta = -scale * zero_point.to(device=scale.device, dtype=scale.dtype).reshape_as(scale)
                self.beta.data.copy_(beta.to(device=self.beta.device, dtype=self.beta.dtype))
            else:
                self.zero_point.data.copy_(
                    zero_point.to(device=self.zero_point.device, dtype=self.zero_point.dtype).reshape_as(self.zero_point)
                )
        self.init_yet = True

    def get_effective_zero_point(self):
        if self.learn_beta:
            safe_scale = self.s.clamp(min=1e-8)
            return -(self.beta / safe_scale)
        return self.zero_point

    def forward(self, x):
        grad_factor = 1.0 / ((max(float(self.thd_pos), 1.0) * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, grad_factor)

        if self.learn_beta:
            beta_scale = grad_scale(self.beta, grad_factor)
            x_norm = (x - beta_scale) / s_scale
        else:
            zp_scale = grad_scale(self.zero_point, grad_factor)
            x_norm = zp_scale + (x / s_scale)

        x_clamped = torch.clamp(x_norm, self.thd_neg, self.thd_pos)

        if self.encode.lower() == "ans":
            self.last_clamped = x_clamped

        x_q = round_pass(x_clamped)
        if self.learn_beta:
            x_q = x_q * s_scale + beta_scale
        else:
            x_q = (x_q - zp_scale) * s_scale
        return x_q

    def get_quant_params(self):
        return self.s, self.get_effective_zero_point(), self.thd_neg < 0


def calcScaleZeroPoint(min_val, max_val, num_bits=8, qmin=None, qmax=None):
    if qmin is None:
        qmin = 0.0
    if qmax is None:
        qmax = 2. ** num_bits - 1.
    scale = (max_val - min_val) / max(float(qmax - qmin), 1.0)
    scale = torch.clamp(scale, min=1e-8)

    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax:
        # zero_point = qmax
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)
    
    zero_point.round_()

    return scale, zero_point


class VanillaQuan(Quantizer):
    def __init__(
        self,
        bit,
        all_positive=True,
        symmetric=False,
        withzeropoint=None,
        encode="deflate",
        channels=1,
        shared_eb=None,
        adaptive_block_quant=False,
    ):
        super().__init__(bit)
        self.all_positive = all_positive
        self.symmetric = symmetric
        self.withzeropoint = all_positive if withzeropoint is None else bool(withzeropoint)
        self.adaptive_block_quant = bool(adaptive_block_quant)
        
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        
        #自适应量化不能限制量化level
        if self.adaptive_block_quant:
            self.thd_neg -= 10000
            self.thd_pos += 10000

        self.bit = bit
        self.register_buffer('scale', torch.tensor([], requires_grad=False))
        self.register_buffer('zero_point', torch.zeros(1, requires_grad=False))
        self.register_buffer('min_val', torch.tensor([], requires_grad=False)) 
        self.register_buffer('max_val', torch.tensor([], requires_grad=False))
        
        # 🌟 新增：编码器选择与熵模型初始化
        self.encode = encode
        if self.encode.lower() == "ans":
            self.entropy_bottleneck = shared_eb
        else:
            self.entropy_bottleneck = None
        self.last_likelihoods = None
        self.last_clamped = None
        
    def update(self, x):
        if self.withzeropoint:
            current_max = x.detach().max()
            current_min = x.detach().min()
            if self.max_val.nelement() == 0 or self.max_val.data < current_max.data:
                self.max_val.data = current_max.data
            if self.min_val.nelement() == 0 or self.min_val.data > current_min.data:
                self.min_val.data = current_min.data
            if self.all_positive:
                self.max_val.clamp_(min=0)
                self.min_val.clamp_(max=0)
            self.max_val.data = torch.maximum(
                self.max_val.data,
                self.min_val.data + self.max_val.new_tensor(1e-8),
            )
            scale, zero_point = calcScaleZeroPoint(
                self.min_val,
                self.max_val,
                self.bit,
                qmin=float(self.thd_neg),
                qmax=float(self.thd_pos),
            )
            self.scale = torch.clamp(scale, min=1e-8)
            self.zero_point.data.copy_(
                zero_point.to(device=self.zero_point.device, dtype=self.zero_point.dtype).reshape_as(self.zero_point)
            )
            return

        if self.all_positive:
            current_max = x.detach().max()
            if self.max_val.nelement() == 0 or self.max_val.data < current_max.data:
                self.max_val.data = current_max.data
            self.max_val.clamp_(min=1e-8)
            self.min_val.data.zero_()
            qmax = max(float(self.thd_pos), 1.0)
            self.scale = torch.clamp(self.max_val / qmax, min=1e-8)
            self.zero_point.data.zero_()
            return

        current_abs_max = x.detach().abs().max()
        if self.max_val.nelement() == 0 or self.max_val.data < current_abs_max.data:
            self.max_val.data = current_abs_max.data    
        self.max_val.clamp_(min=1e-8)
        self.min_val.data = -self.max_val.data
        qmax = max(float(self.thd_pos), 1.0)
        self.scale = torch.clamp(self.max_val / qmax, min=1e-8)

        # 原始的vanilla量化，非对称有zeropoint
        # if self.max_val.nelement() == 0 or self.max_val.data < x.max().data:
        #     self.max_val.data = x.max().data
        # self.max_val.clamp_(min=0)
        
        # if self.min_val.nelement() == 0 or self.min_val.data > x.min().data:
        #     self.min_val.data = x.min().data 
        # self.min_val.clamp_(max=0)    
        
        # self.scale, self.zero_point = calcScaleZeroPoint(self.min_val, self.max_val, self.bit)
        self.zero_point.data.zero_()
    
    def forward(self, x):
        self.update(x)
        
        if self.withzeropoint:
            x_norm = self.zero_point + (x / self.scale)
        else:
            x_norm = x / self.scale
        x_clamped = torch.clamp(x_norm, self.thd_neg, self.thd_pos)
        
        # 2. 🌟 新增：缓存 clamped 值供外部批处理
        if self.encode.lower() == "ans":
            # _, likelihoods = self.entropy_bottleneck(x_clamped.view(1, 1, -1, 1))
            # self.last_likelihoods = likelihoods.view(x.shape)
            self.last_clamped = x_clamped
            
        # 3. 离散化与反量化
        x_q = round_pass(x_clamped)
        if self.withzeropoint:
            x_q = self.scale * (x_q - self.zero_point)
        else:
            x_q = self.scale * x_q
        return x_q

    def get_quant_params(self):
        return self.scale, self.zero_point, self.thd_neg < 0
