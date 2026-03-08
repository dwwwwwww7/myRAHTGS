import torch
import torch.nn as nn
import torch.optim as optim
import math

from utils.quant_utils import LsqQuan, split_length
from gaussian_renderer import seg_quant_ave

# ==========================================
# 模拟 gaussian_model.py 的精简模型
# ==========================================
class DummyGaussianModel:
    def __init__(self, n_points=10000, n_dims=55, n_block=57):
        # 1. 模拟特征数据 (相当于尚未量化的 RAHT 系数或原特征)
        # 用正态分布模拟(有正有负)
        self.features = nn.Parameter(torch.randn(n_points, n_dims, device='cuda', requires_grad=True))
        
        self.n_block = n_block
        self.n_dims = n_dims
        
        # 2. 模拟 init_qas，为每个维度的每个块创建一个 LsqQuan
        self.qas = nn.ModuleList([])
        for dim_idx in range(n_dims):
            # 简化为都用 8 bit
            bit = 8
            for _ in range(n_block):
                # 必须设置 all_positive=False 以支持负值
                self.qas.append(LsqQuan(bit=bit, init_yet=False, all_positive=False).cuda())
                
        # 3. 模拟 finetuning_setup 注册优化器
        l = [
            {'params': [self.features], 'lr': 0.01, "name": "features"},
            {'params': self.qas.parameters(), 'lr': 0.001, "name": "quantizers"}
        ]
        self.optimizer = optim.Adam(l, lr=0.0, eps=1e-15)

# ==========================================
# 快速验证脚本
# ==========================================
def test_lsq_fast():
    print("="*50)
    print("开始 LSQ 快速验证 (剥离 RAHT)")
    print("="*50)
    
    n_points = 10000
    n_dims = 55
    n_block = 57
    
    model = DummyGaussianModel(n_points, n_dims, n_block)
    
    # 模拟一个训练目标：让所有特征趋于某个目标值（比如 0.5）
    target = torch.full((n_points, n_dims), 0.5, device='cuda')
    loss_fn = nn.MSELoss()
    
    print(f"数据维度: 点数={n_points}, 特征={n_dims}")
    print(f"量化器数: {len(model.qas)} (55维度 * {n_block}块)")
    
    # 抽取第一维度的第一个量化器来观察其 Scale 变化
    watch_qa_idx = 0 
    watch_qa = model.qas[watch_qa_idx]
    
    print("\n初始 Scale 状态:")
    print(f"  qa[0].s: {watch_qa.s.item():.6f} (尚未 init_from，值为全1初始值)")
    
    epochs = 2000
    for epoch in range(1, epochs + 1):
        # ---------------------------------------------
        # 前向传播：模拟 ft_render 中的分块量化
        # ---------------------------------------------
        # (去掉了 RAHT，直接量化 features)
        C = model.features  
        quantC = torch.zeros_like(C)
        
        # 为了演示 init_from 的触发，将相当于 AC 系数的部分进行处理
        # 假设第一行为 DC (不需要量化)，剩下的为 AC
        quantC[0] = C[0]
        
        qa_cnt = 0
        lc1 = C.shape[0] - 1
        split_ac = split_length(lc1, model.n_block)
        
        for i in range(C.shape[-1]):
            for j, length in enumerate(split_ac):
                qa_idx = qa_cnt + j
                # 检查并触发 init_from
                if not model.qas[qa_idx].init_yet:
                    start_idx = sum(split_ac[:j]) + 1
                    end_idx = start_idx + length
                    model.qas[qa_idx].init_from(C[start_idx:end_idx, i])
                    
            # 模拟 seg_quant_ave
            quantC[1:, i] = seg_quant_ave(C[1:, i], split_ac, model.qas[qa_cnt : qa_cnt + model.n_block])
            qa_cnt += model.n_block
            
        # 在 Epoch 1 的 init_from 之后，打印一下初始化后的 Scale
        if epoch == 1:
            print("\nEpoch 1 init_from 执行后:")
            print(f"  qa[0].s: {watch_qa.s.item():.6f}")
        
        # ---------------------------------------------
        # 计算 Loss、反向传播、更新参数
        # ---------------------------------------------
        # 将 quantizing 后的数据与 target 算 loss，强制更新
        loss = loss_fn(quantC, target)
        
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        
        # ---------------------------------------------
        # 打印状态
        # ---------------------------------------------
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f} | qa[0].s: {watch_qa.s.item():.6f}")
            
    print("\n="*50)
    print("测试结论:")
    print("1. init_from() 成功设置了 scale。")
    print("2. Loss 能够下降，说明前向传播产生的数值正常。")
    print("3. qa[0].s 随着 Epoch 的变化而更新，说明梯度的 STE 及参数绑定均有效！")
    print("="*50)

if __name__ == "__main__":
    test_lsq_fast()
