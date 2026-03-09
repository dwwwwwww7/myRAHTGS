import math

import torch
import torch.nn as nn
from torch.autograd import Function

# CompressAI 熵模型（ECSQ 量化器依赖）
try:
    from compressai.entropy_models import EntropyBottleneck
    COMPRESSAI_AVAILABLE = True
except ImportError:
    COMPRESSAI_AVAILABLE = False
    print("[WARNING] compressai 未安装，EcsqQuan 量化器不可用。请运行: pip install compressai")

def split_length(length, n):
    base_length = length / n
    floor_length = int(base_length)
    remainder = length - (floor_length * n)
    result = [floor_length + 1] * remainder + [floor_length] * (n - remainder)
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

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError


class IdentityQuan(Quantizer):
    def __init__(self, bit=None, *args, **kwargs):
        super().__init__(bit)
        assert bit is None, 'The bit-width of identity quantizer must be None'

    def forward(self, x):
        return x


class LsqQuan(Quantizer):
    def __init__(self, bit, init_yet, all_positive=True, symmetric=False):
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

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x


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
    def __init__(self, bit, all_positive=True, symmetric=False):
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

        self.bit = bit
        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min_val = torch.tensor([], requires_grad=False)
        max_val = torch.tensor([], requires_grad=False)
        
        self.register_buffer('scale', scale)
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('min_val', min_val) 
        self.register_buffer('max_val', max_val)
        
    def update(self, x):
        if self.max_val.nelement() == 0 or self.max_val.data < x.max().data:
            self.max_val.data = x.max().data
        self.max_val.clamp_(min=0)
        
        if self.min_val.nelement() == 0 or self.min_val.data > x.min().data:
            self.min_val.data = x.min().data 
        self.min_val.clamp_(max=0)    
        
        self.scale, self.zero_point = calcScaleZeroPoint(self.min_val, self.max_val, self.bit)
    
    def forward(self, x):
        self.update(x)
        x = self.zero_point + (x / self.scale)
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = self.scale * (x - self.zero_point)
        return x


# ============================================================================
# ECSQ - 熵约束标量量化器 (Entropy-Constrained Scalar Quantization)
# ============================================================================
#
# 原理：
#   与 LSQ/Vanilla 等固定位宽量化器不同，ECSQ 不预设量化位数，
#   而是通过联合优化 Rate-Distortion (R-D) 损失来自适应分配码率。
#
#   - Training 阶段：用 [-0.5, 0.5] 均匀噪声替代 Round，实现梯度完美反传
#   - Inference 阶段：恢复真正的离散化 Round
#   - 熵模型 (EntropyBottleneck) 会动态拟合量化后系数的 PDF，
#     精确输出每个系数的 -log2(P)，用于 Rate Loss 的计算
#   - 可学习步长 s 在 R-D 优化下自适应调整：
#     对渲染贡献小的高频 AC 系数 → s 增大 → 量化后大面积归零 → 极度稀疏
#
# 使用方式：
#   量化器 forward 返回反量化后的张量（与 LSQ/Vanilla 接口一致），
#   同时将 likelihoods 暂存在 self.last_likelihoods 中，
#   供训练循环在 forward 结束后提取并计算 Rate Loss。
# ============================================================================

class EcsqQuan(Quantizer):
    """
    熵约束可学习标量量化器 (ECSQ)
    
    不同于 LsqQuan/VanillaQuan 使用固定位宽，ECSQ 通过 CompressAI 的
    EntropyBottleneck 动态拟合量化后系数的概率分布，用 R-D Loss 联合优化码率。
    
    Args:
        channels: 熵模型的通道数（对应每个量化器处理的特征通道数，通常为1）
        init_step_size: 初始步长
    """
    def __init__(self, channels=1, init_step_size=1.0):
        super().__init__(bit=None)  # ECSQ 不受限于固定 bit，依靠熵模型动态优化
        
        if not COMPRESSAI_AVAILABLE:
            raise RuntimeError(
                "EcsqQuan 需要 compressai 库。请运行: pip install compressai"
            )
        
        # 核心概率模型：学习量化后系数的概率密度函数 (PDF)
        self.entropy_bottleneck = EntropyBottleneck(channels)
        
        # 可学习的步长参数 s，形状 (1, channels)
        # 在 R-D 优化下，对渲染贡献小的系数的 s 会自动增大，
        # 使得量化后大面积归零，达到极度稀疏化
        self.s = nn.Parameter(torch.ones(1, channels) * init_step_size)
        self.min_step_size = 1e-4
        self.init_yet = False
        
        # 暂存前向传播产生的 likelihoods（概率），以免破坏原有仅返回 x 的 API
        # 训练循环在 forward 结束后可通过 qa.last_likelihoods 提取
        self.last_likelihoods = None

    def init_from(self, x, *args, **kwargs):
        """
        根据输入数据的分布初始化步长（类似 LSQ 的初始化策略）
        """
        with torch.no_grad():
            self.s.data.fill_(x.detach().abs().mean().item() * 2.0)
        self.init_yet = True

    def forward(self, x):
        """
        前向传播：
          1. 将输入除以步长 s → 归一化
          2. reshape 为 CompressAI 需要的 4D 格式 (B, C, H, W)
          3. 送入 EntropyBottleneck：
             - Training: 加均匀噪声模拟量化
             - Inference: 真正 Round
          4. 乘回步长 s → 反量化
          5. 暂存 likelihoods 供 Rate Loss 计算
        
        Args:
            x: 输入的 AC 系数，形状为 (N,) 的一维张量
        
        Returns:
            x_q: 反量化后的张量，形状与输入一致（与 LSQ/Vanilla 接口兼容）
        """
        # 限制步长下界，防止除零
        s_clamped = self.s.clamp(min=self.min_step_size)
        
        # 保存原始形状
        orig_shape = x.shape
        
        # 归一化：除以步长
        y = (x / s_clamped).view(1, 1, -1, 1)  # reshape 为 (B=1, C=1, H=N, W=1)
        
        # 送入 EntropyBottleneck
        # Training 时：y_hat = y + uniform_noise (连续松弛)
        # Inference 时：y_hat = round(y) (真正离散量化)
        # likelihoods：每个元素被量化后的概率 P(y_hat)
        y_hat, likelihoods = self.entropy_bottleneck(y)
        
        # 反量化：乘回步长，并恢复原始形状
        x_q = (y_hat * s_clamped).view(orig_shape)
        
        # 将概率保存在模块内部，供后续计算 Rate Loss 时提取
        # Rate Loss = -log2(likelihoods).sum() / num_points
        self.last_likelihoods = likelihoods.view(orig_shape)
        
        return x_q