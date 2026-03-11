import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 把当前工作目录加入 sys.path 以便导入 scene模块 的函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from scene.gaussian_model import unpack_bits
except ImportError:
    print("无法从 scene.gaussian_model 导入 unpack_bits，请确保在项目根目录下运行此脚本。")
    sys.exit(1)

def analyze_orgb(npz_path):
    print(f"=========================================")
    print(f"正在读取文件: {npz_path}")
    print(f"=========================================")
    
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"加载 npz 文件失败: {e}")
        return
        
    print("文件中包含的键 (Keys):", data.files)
    
    # 提取并验证 DC 系数 
    if 'f' in data:
        f_dc = data['f']
        print(f"DC 根系数形状: {f_dc.shape} (通常是 55 维)")
    else:
        print("警告: 找不到键 'f'，这可能不是一个标准的 orgb.npz 文件。")
    
    ac_data = None
    
    # 判断存储的具体格式
    if 'packed' in data:
        print("\n检测到: [位打包 (Bit-Packed) 格式]")
        bitstream = data['i'].tobytes()
        bit_config = data['bit_config']
        signed_config = data['signed_config']
        
        # 反推 N 的大小（AC 系数的数量）
        # 位打包时物理字节可能在末尾有几个 pad 的 bit，所以用整除即可
        total_bits = len(bitstream) * 8
        sum_bits = sum(bit_config)
        N = total_bits // sum_bits
        
        print(f"  推断的高频 AC 系数数量 (N): {N}")
        print(f"  各维度位宽配置 (总和 {sum_bits} bits): \n  {bit_config}")
        print(f"  是否包含有符号(二补码)位数: {any(signed_config)}")
        
        print("\n正在解包数据 (这可能需要花费数十秒时间，请耐心等待) ...")
        
        # 【关键修改】：为了查看实际上被 ZIP 压缩的补码（即跨字节打断前的全真二进制数值），
        # 我们在这里故意将 signed_config 全部覆盖为 False。
        # 这样 unpack_bits 解析出来的 -1（比如 5-bit 下）就会保持为 31（即全 1），
        # 这正是底层打包字节流中真实记录、并且被 ZIP 尝试强行压缩的数据样子。
        fake_unsigned_config = np.zeros_like(signed_config, dtype=bool) 
        ac_data = unpack_bits(bitstream, bit_config, N, fake_unsigned_config)
        
    elif 'i' in data and 'packed' not in data:
        print("\n检测到: [无压缩或兼容 (Uint8/Uint16) 格式]")
        ac_data = data['i']
        
    elif 'ecsq_meta' in data:
        print("\n检测到: [ECSQ 算术编码格式]")
        print("由于 ECSQ 进行了基于 EntropyBottleneck 的极限压缩，需要在前向传播中借助模型进行算术解码。")
        print("本脚本暂时只支持 Vanilla 和 LSQ 生成的纯量化/位打包数据可视化分析。")
        return
    else:
        print("\n未能识别数据格式或缺少关键的高频系数组 'i'。")
        return
        
    print(f"\n成功获取 AC 高频系数特征。其形状为: {ac_data.shape}")
    
    # 定义 55 维属性中不同成分的切片范围
    # 0: opacity (1)
    # 1-3: euler (3)
    # 4-6: f_dc (3)
    # 7-15: f_rest_0 (9)
    # 16-30: f_rest_1 (15)
    # 31-51: f_rest_2 (21)
    # 52-54: scale (3)
    
    attr_slices = {
        'Opacity': slice(0, 1),
        'Euler Angles': slice(1, 4),
        'Features DC': slice(4, 7),
        'Features Rest (SH Deg 1)': slice(7, 16),
        'Features Rest (SH Deg 2)': slice(16, 31),
        'Features Rest (SH Deg 3)': slice(31, 52),
        'Scale': slice(52, 55)
    }
    
    # 开始绘图
    print("正在绘制数值分布直方图...")
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle(f"Quantized AC Coefficients Distribution\n{os.path.basename(npz_path)}", fontsize=16, y=0.98)
    
    for idx, (name, slc) in enumerate(attr_slices.items()):
        ax = axes[idx]
        # 提取相关维度的数据并展平成一维数组
        vals = ac_data[:, slc].flatten()
        
        # 为了更清楚地看到分布规律，这里使用 log 刻度
        # 如果你希望看线性频率，可以去除 log=True
        ax.hist(vals, bins=50, alpha=0.7, color='steelblue', edgecolor='black', log=True)
        
        # 设置标题和标签
        ax.set_title(f"{name} (Dims {slc.start}-{slc.stop-1})")
        ax.set_xlabel("Quantized Value")
        ax.set_ylabel("Frequency (Log Scale)")
        ax.grid(axis='y', alpha=0.3)
        
    # 隐藏用不到的剩余子图
    for i in range(len(attr_slices), len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存并展示图片
    output_png = "orgb_distribution.png"
    plt.savefig(output_png, dpi=300)
    print(f"\n✅ 绘图完成！分布图已保存至项目根目录下的: '{output_png}'")
    
    # 尝试在窗口中显示
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    # -----------------------------------------------------------------
    # 【请在这里手动填写你要分析的 orgb.npz 文件的绝对路径或相对路径】
    # -----------------------------------------------------------------
    ORGB_PATH = "/data/zdw/myRAHT2026/myRAHTGS/results_2026/lsq0311/train_config4/point_cloud/iteration_0/pc_npz/bins/orgb.npz" 
    
    if ORGB_PATH == "example_exp_dir/bins/orgb.npz" or not os.path.exists(ORGB_PATH):
        print(f"⚠️ 请使用代码编辑器打开此脚本，并将 ORGB_PATH 修改为实际存在的 orgb.npz 文件路径。")
        print(f"当前填写的路径: {ORGB_PATH}")
    else:
        analyze_orgb(ORGB_PATH)
