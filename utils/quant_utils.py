import math

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


def _laplace_cdf(x, scale):
    """Numerically stable Laplace CDF.

    Uses exp(-|x|/scale) in both branches to avoid the classic
    torch.where + exp overflow:  when x > 0 the old ``exp(x/scale)``
    overflows to Inf, and the backward of ``torch.where`` computes
    ``0 * Inf = NaN``.  Using ``exp(-|x|/scale)`` (exponent always ≤ 0)
    guarantees no overflow while remaining mathematically equivalent.
    """
    safe_scale = scale.clamp(min=1e-6)
    abs_x = x.abs()
    exp_term = torch.exp(-abs_x / safe_scale)
    # x >= 0 : CDF = 1 - 0.5 * exp(-x/b)
    # x <  0 : CDF = 0.5 * exp(-|x|/b)
    return torch.where(x >= 0, 1.0 - 0.5 * exp_term, 0.5 * exp_term)


def _estimate_zero_mean_laplace_bits(symbols, split_tensor):
    block_lengths = split_tensor.to(dtype=symbols.dtype).view(1, -1, 1)
    max_block_len = symbols.shape[-1]
    valid_mask = (
        torch.arange(max_block_len, device=symbols.device, dtype=torch.long)
        .view(1, 1, -1)
        < split_tensor.view(1, -1, 1)
    )
    valid_mask_f = valid_mask.to(dtype=symbols.dtype)

    # Zero-mean Laplace MLE: b = E[|x|], detached.
    # raw_scale: 仅做数值安全下界（1e-6），供压缩路径使用，不影响实际分布。
    # train_scale: 训练专用，加 0.01 下界防止分布极尖导致梯度爆炸。
    raw_scale = (
        (symbols.detach().abs() * valid_mask_f).sum(dim=-1)
        / block_lengths.squeeze(-1).clamp(min=1.0)
    ).clamp(min=1e-6)

    # 【保护1（仅训练）】scale 下界 0.01：只用于 bits 估计，不返回给压缩路径
    train_scale = raw_scale.clamp(min=0.01).unsqueeze(-1)  # [n_ch, n_blocks, 1]

    # 直接使用符号计算 CDF。由于 _laplace_cdf 已使用 exp(-|x|) 不会溢出，
    # 我们可以安全地计算真实概率，不会产生 NaN。
    upper = _laplace_cdf(symbols + 0.5, train_scale)
    lower = _laplace_cdf(symbols - 0.5, train_scale)

    # 【保护2】likelihood 下界 1e-9：最大单元素罚项 -log2(1e-9) ≈ 29.9 bits
    # 既防止 log2(0) 产生 NaN，又保留了足够的 bits 空间评估长尾（如 x=100）。
    probs = (upper - lower).clamp(min=1e-9)
    per_element_bits = -torch.log2(probs)
    total_bits = (per_element_bits * valid_mask_f).sum()
    # 返回原始 MLE scale（无 0.01 截断），供 estimate_zero_mean_laplace_block_scales 复用
    return total_bits, raw_scale


def _laplace_bits_with_tail_clip(quantized_symbols, block_scale, sigma=3.0):
    """Compute -log2(prob) per symbol using input-clamp + probability-floor.

    Numerics follow the same three-layer protection as
    ``_estimate_zero_mean_laplace_bits``:

    1. ``block_scale`` lower-bounded at 0.01 (caller's responsibility, but
       also re-applied here for safety) → prevents an extremely peaked
       distribution where the CDF difference collapses.
    2. Input 3-sigma clamp *before* the CDF: ``x ← clamp(x, -σb, σb)``.
       Outlier symbols are pulled to the boundary so their CDF interval
       probability remains a reasonable positive number and the gradient
       still flows through the symbol value.
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

    upper = _laplace_cdf(quantized_symbols + 0.5, safe_scale)
    lower = _laplace_cdf(quantized_symbols - 0.5, safe_scale)

    # 【保护2】likelihood 下界 1e-9 (最大罚 ~29.9 bits)
    probs = (upper - lower).clamp(min=1e-9)
    return -torch.log2(probs)


def estimate_zero_mean_laplace_block_scales(symbols, split):
    """压缩路径专用：计算零均值 Laplace MLE 尺度参数 b = E[|x|]。

    下界仅为数值安全的 1e-6，完全不受训练保护（0.01下界/3σ截断/1e-5 floor）影响，
    保证熵编码 CDF 忠实反映真实数据分布，不影响实际压缩效率。
    """
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

    # 独立计算 MLE scale，不经过任何训练保护逻辑
    block_lengths = split_tensor.to(dtype=symbols.dtype).view(1, -1, 1)
    valid_mask_f = (
        torch.arange(max_block_len, device=symbols.device, dtype=torch.long)
        .view(1, 1, -1)
        < split_tensor.view(1, -1, 1)
    ).to(dtype=symbols.dtype)
    block_scales = (
        (packed.detach().abs() * valid_mask_f).sum(dim=-1)
        / block_lengths.squeeze(-1).clamp(min=1.0)
    ).clamp(min=1e-6)  # 仅数值安全，不影响分布形状

    if squeeze_output:
        return block_scales.reshape(-1)
    return block_scales


def batched_quantize_blocks(x, split, qas, return_symbols=False, return_trans=False, return_ans_bits=False):
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

    flat_qas = list(qas)
    n_points, n_dims = x.shape
    n_blocks = len(split)
    expected_qas = n_dims * n_blocks
    if len(flat_qas) != expected_qas:
        raise ValueError(f"Expected {expected_qas} quantizers, got {len(flat_qas)}")

    split_tensor, block_ids, local_ids = _build_block_index_tensors(split, x.device)
    if int(split_tensor.sum().item()) != n_points:
        raise ValueError(f"Split sum {int(split_tensor.sum().item())} does not match input length {n_points}")

    max_block_len = int(split_tensor.max().item())
    xt = x.transpose(0, 1).contiguous()
    packed = xt.new_zeros((n_dims, n_blocks, max_block_len))
    packed[:, block_ids, local_ids] = xt

    abs_packed = packed.abs()
    block_lengths = split_tensor.to(dtype=x.dtype).view(1, n_blocks)

    first_qa = flat_qas[0]
    if hasattr(first_qa, "init_yet"):
        block_abs_mean = abs_packed.sum(dim=-1) / block_lengths
        for dim_idx in range(n_dims):
            base = dim_idx * n_blocks
            for block_idx in range(n_blocks):
                qa = flat_qas[base + block_idx]
                if not qa.init_yet:
                    init_scale = block_abs_mean[dim_idx, block_idx] * 2 / (float(qa.thd_pos) ** 0.5)
                    with torch.no_grad():
                        qa.s.data.fill_(init_scale.item())
                    qa.init_yet = True
    elif hasattr(first_qa, "scale"):
        block_abs_max = abs_packed.amax(dim=-1)
        block_max = packed.amax(dim=-1)
        for dim_idx in range(n_dims):
            base = dim_idx * n_blocks
            for block_idx in range(n_blocks):
                qa = flat_qas[base + block_idx]
                with torch.no_grad():
                    if qa.all_positive:
                        current_max = block_max[dim_idx, block_idx].detach()
                        if qa.max_val.nelement() == 0 or qa.max_val.data < current_max.data:
                            qa.max_val.data = current_max.data
                        qa.max_val.clamp_(min=0)
                        qa.min_val.data = torch.zeros_like(qa.max_val.data)
                        qmax = max(float(qa.thd_pos), 1.0)
                        qa.scale = torch.clamp(qa.max_val / qmax, min=1e-8)
                    else:
                        current_abs_max = block_abs_max[dim_idx, block_idx].detach()
                        if qa.max_val.nelement() == 0 or qa.max_val.data < current_abs_max.data:
                            qa.max_val.data = current_abs_max.data
                        qa.max_val.clamp_(min=1e-8)
                        qa.min_val.data = -qa.max_val.data
                        qmax = max(float(qa.thd_pos), 1.0)
                        qa.scale = torch.clamp(qa.max_val / qmax, min=1e-8)
                    qa.zero_point.data.zero_()
    else:
        raise ValueError(f"Unsupported quantizer type: {type(first_qa)}")

    scale_entries = []
    neg_entries = []
    pos_entries = []
    for qa in flat_qas:
        if hasattr(qa, "s"):
            scale_entries.append(qa.s.reshape(1))
        else:
            scale_entries.append(qa.scale.reshape(1))
        neg_entries.append(scale_entries[-1].new_tensor(float(qa.thd_neg)))
        pos_entries.append(scale_entries[-1].new_tensor(float(qa.thd_pos)))

    scales = torch.stack(scale_entries, dim=0).view(n_dims, n_blocks, 1)
    thd_neg = torch.stack(neg_entries, dim=0).view(n_dims, n_blocks, 1)
    thd_pos = torch.stack(pos_entries, dim=0).view(n_dims, n_blocks, 1)

    if hasattr(first_qa, "s"):
        grad_factors = 1.0 / torch.sqrt(thd_pos.clamp(min=1.0) * block_lengths.view(1, n_blocks, 1))
        scales = grad_scale(scales, grad_factors)

    x_clamped = torch.clamp(packed / scales, thd_neg, thd_pos)
    x_q = round_pass(x_clamped)
    dequantized = x_q * scales

    flat_clamped = x_clamped[:, block_ids, local_ids]
    flat_symbols = x_q[:, block_ids, local_ids].transpose(0, 1).contiguous()
    flat_dequantized = dequantized[:, block_ids, local_ids].transpose(0, 1).contiguous()

    ans_bits = x.new_zeros(())
    entropy_scales = None
    encode_mode = getattr(first_qa, "encode", "deflate").lower()
    needs_clamped_cache = any(getattr(qa, "encode", "deflate").lower() == "ans" for qa in flat_qas)
    if return_ans_bits and encode_mode == "laplace":
        ans_bits, entropy_scales = _estimate_zero_mean_laplace_bits(x_q, split_tensor)
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

    trans = None
    if return_trans:
        trans = []
        for qa in flat_qas:
            i_scale, i_zp, _ = qa.get_quant_params()
            trans.extend([i_scale.item(), i_zp.item()])

    if squeeze_output:
        flat_dequantized = flat_dequantized.reshape(-1)
        flat_symbols = flat_symbols.reshape(-1)

    if return_symbols and return_trans and return_ans_bits:
        return flat_dequantized, flat_symbols, trans, ans_bits
    if return_symbols and return_trans:
        return flat_dequantized, flat_symbols, trans
    if return_symbols and return_ans_bits:
        return flat_dequantized, flat_symbols, ans_bits
    if return_symbols:
        return flat_dequantized, flat_symbols
    if return_trans and return_ans_bits:
        return flat_dequantized, trans, ans_bits
    if return_trans:
        return flat_dequantized, trans
    if return_ans_bits:
        return flat_dequantized, ans_bits
    return flat_dequantized

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
            # LSQ论文里的初始化方法
            self.s.data.fill_(x.detach().abs().mean().item() * 2 / (self.thd_pos ** 0.5))
            # 以下为min-max初始化
            # min_val = x.detach().min()
            # max_val = x.detach().max()
            # if max_val == min_val:
            #     max_val = max_val + 1e-5
            # scale_init = (max_val - min_val) / (self.thd_pos - self.thd_neg)
            # self.s.data.fill_(scale_init.item())
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


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax:
        # zero_point = qmax
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)
    
    zero_point.round_()

    return scale, zero_point


class VanillaQuan(Quantizer):
    def __init__(self, bit, all_positive=True, symmetric=False, encode="deflate", channels=1, shared_eb=None):
        super().__init__(bit)
        self.all_positive = all_positive
        self.symmetric = symmetric
        
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
        # Old affine min-max quantization kept for reference. It uses a
        # block-specific zero_point and often creates multi-peak symbol
        # distributions after grouping multiple blocks together.
        #
        # if self.max_val.nelement() == 0 or self.max_val.data < x.max().data:
        #     self.max_val.data = x.max().data
        # self.max_val.clamp_(min=0)
        #
        # if self.min_val.nelement() == 0 or self.min_val.data > x.min().data:
        #     self.min_val.data = x.min().data
        # self.min_val.clamp_(max=0)
        #
        # self.scale, self.zero_point = calcScaleZeroPoint(self.min_val, self.max_val, self.bit)

        # Vanilla quantization now uses a fixed zero-centered symbol
        # definition. The old affine min-max path is intentionally kept
        # commented above for reference.
        if self.all_positive:
            current_max = x.detach().max()
            if self.max_val.nelement() == 0 or self.max_val.data < current_max.data:
                self.max_val.data = current_max.data
            self.max_val.clamp_(min=0)
            self.min_val.data = torch.zeros_like(self.max_val.data)
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
        self.zero_point.data.zero_()
    
    def forward(self, x):
        self.update(x)
        
        # 1. 归一化与截断
        # Old affine path kept for reference:
        # x_norm = self.zero_point + (x / self.scale)
        x_norm = x / self.scale
        x_clamped = torch.clamp(x_norm, self.thd_neg, self.thd_pos)
        
        # 2. 🌟 新增：缓存 clamped 值供外部批处理
        if self.encode.lower() == "ans":
            # _, likelihoods = self.entropy_bottleneck(x_clamped.view(1, 1, -1, 1))
            # self.last_likelihoods = likelihoods.view(x.shape)
            self.last_clamped = x_clamped
            
        # 3. 离散化与反量化
        x_q = round_pass(x_clamped)
        # Old affine path kept for reference:
        # x_q = self.scale * (x_q - self.zero_point)
        x_q = self.scale * x_q
        return x_q

    def get_quant_params(self):
        return self.scale, self.zero_point, self.thd_neg < 0
