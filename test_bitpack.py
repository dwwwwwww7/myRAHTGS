import torch
import numpy as np
from scene.gaussian_model import pack_bits, unpack_bits

print("=== 位打包功能测试 ===")

# 测试 1: 基本功能 (包含有符号数)
print("\n测试 1: 基本功能")
# 注意：16位有符号数的范围是 -32768 到 32767，这里用 -30000 进行测试
# 9位有符号数的范围是 -256 到 255，这里用 -200 进行测试
arr = torch.tensor([[1, -200, 25, -30000]])  # 转换为 (1, 4) 形状
bit_widths = [3, 9, 6, 16] 
signed_flags = [False, True, False, True] # 设置第2个和第4个通道为有符号数

print(f"原始数据: {arr}")
print(f"位宽配置: {bit_widths}")
print(f"符号标志: {signed_flags}")

packed = pack_bits(arr, bit_widths)
print(f"打包结果: {len(packed)} bytes")
print(f"二进制形式: {' '.join(f'{b:08b}' for b in packed)}")
print(f"十进制形式: {list(packed)}")

unpacked = unpack_bits(packed, bit_widths, arr.shape[0], signed_flags)
print(f"解包结果: {unpacked}")
print(f"数据正确性: {'✓' if torch.allclose(arr.float(), torch.from_numpy(unpacked).float()) else '✗'}")

# 测试 2: 多维数据 (包含有符号数)
print("\n测试 2: 多维数据")
# 4位有符号数范围: -8 到 7
# 9位有符号数范围: -256 到 255
# 6位有符号数范围: -32 到 31
data = torch.tensor([[1, -250, 25], [-7, 200, -31], [7, -150, 31]])
bit_depths = [4, 9, 6]
signed_multi = [True, True, True] # 将所有列设为有符号数来进行测试

print(f"原始数据形状: {data.shape}")
print(f"位宽配置: {bit_depths}")
print(f"符号标志: {signed_multi}")

packed_multi = pack_bits(data, bit_depths)
unpacked_multi = unpack_bits(packed_multi, bit_depths, data.shape[0], signed_multi)
unpacked_reshaped = unpacked_multi

print(f"打包大小: {len(packed_multi)} bytes")
# 由于二进制和十进制打印比较长，这里省略，仅验证正确性
print(f"数据正确性: {'✓' if torch.allclose(data.float(), torch.from_numpy(unpacked_reshaped).float()) else '✗'}")


print("\n=== 测试完成 ===")