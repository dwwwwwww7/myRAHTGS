#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 万皓师兄基于mesongs的raht修改的，封装了一下
"""
Created on May 25, 2021
Modified on Jul 20, 2024

This code is derived by the implementation of 3DAC. See https://fatpeter.github.io/ for more details.

It is an python version of RAHT based on https://github.com/digitalivp/RAHT/tree/reorder.
The original C implementation is more readable.

"""
import numpy as np
import torch


# ============================================================
# Morton 编码 / 解码相关工具函数
# Morton 编码（Z-order curve）将 3D 坐标 (x, y, z) 交错编码为一维整数，
# 使得空间中相邻的点在一维排序后也倾向于相邻，便于后续 RAHT 变换。
# ============================================================

def copyAsort(V):
    # input
    # V: np.array (n,3), input vertices

    # output
    # W: np.array (n,), weight
    # val: np.array (n,), zyx val of vertices
    # reord: np.array (n,), idx ord after sort



    V=V.astype(np.uint64)

    # 叶节点权重初始化为 1
    W=np.ones(V.shape[0])

    # 将 zyx 坐标的每一位交错排列，生成 Morton 码
    # 提取各轴坐标（注意输入顺序是 z,y,x → vx=x, vy=y, vz=z）
    vx, vy, vz= V[:,2], V[:,1], V[:,0]
    val = ((0x000001 & vx)    ) + ((0x000001 & vy)<< 1) + ((0x000001 &  vz)<< 2) + \
                ((0x000002 & vx)<< 2) + ((0x000002 & vy)<< 3) + ((0x000002 &  vz)<< 4) + \
                ((0x000004 & vx)<< 4) + ((0x000004 & vy)<< 5) + ((0x000004 &  vz)<< 6) + \
                ((0x000008 & vx)<< 6) + ((0x000008 & vy)<< 7) + ((0x000008 &  vz)<< 8) + \
                ((0x000010 & vx)<< 8) + ((0x000010 & vy)<< 9) + ((0x000010 &  vz)<<10) + \
                ((0x000020 & vx)<<10) + ((0x000020 & vy)<<11) + ((0x000020 &  vz)<<12) + \
                ((0x000040 & vx)<<12) + ((0x000040 & vy)<<13) + ((0x000040 &  vz)<<14) + \
                ((0x000080 & vx)<<14) + ((0x000080 & vy)<<15) + ((0x000080 &  vz)<<16) + \
                ((0x000100 & vx)<<16) + ((0x000100 & vy)<<17) + ((0x000100 &  vz)<<18) + \
                ((0x000200 & vx)<<18) + ((0x000200 & vy)<<19) + ((0x000200 &  vz)<<20) + \
                ((0x000400 & vx)<<20) + ((0x000400 & vy)<<21) + ((0x000400 &  vz)<<22) + \
                ((0x000800 & vx)<<22) + ((0x000800 & vy)<<23) + ((0x000800 &  vz)<<24) + \
                ((0x001000 & vx)<<24) + ((0x001000 & vy)<<25) + ((0x001000 &  vz)<<26) + \
                ((0x002000 & vx)<<26) + ((0x002000 & vy)<<27) + ((0x002000 &  vz)<<28) + \
                ((0x004000 & vx)<<28) + ((0x004000 & vy)<<29) + ((0x004000 &  vz)<<30) + \
                ((0x008000 & vx)<<30) + ((0x008000 & vy)<<31) + ((0x008000 &  vz)<<32) + \
                ((0x010000 & vx)<<32) + ((0x010000 & vy)<<33) + ((0x010000 &  vz)<<34) + \
                ((0x020000 & vx)<<34) + ((0x020000 & vy)<<35) + ((0x020000 &  vz)<<36) + \
                ((0x040000 & vx)<<36) + ((0x040000 & vy)<<37) + ((0x040000 &  vz)<<38) + \
                ((0x080000 & vx)<<38) + ((0x080000 & vy)<<39) + ((0x080000 &  vz)<<40)
    # 上面每一位交错编码共 20 位（支持到 2^20=1048576 范围的坐标）
    # 被注释掉的更高位（21-24 位）可根据需要启用
    # + \
                # ((0x100000 & vx)<<40) + ((0x100000 & vy)<<41) + ((0x100000 &  vz)<<42) + \
                # ((0x200000 & vx)<<42) + ((0x200000 & vy)<<43) + ((0x200000 &  vz)<<44) + \
                # ((0x400000 & vx)<<44) + ((0x400000 & vy)<<45) + ((0x400000 &  vz)<<46) + \
                # ((0x800000 & vx)<<46) + ((0x800000 & vy)<<47) + ((0x800000 &  vz)<<48)

    # 按 Morton 码升序排序，返回排序索引
    reord=np.argsort(val)
    val=np.sort(val)
    val = val.astype(np.uint64)
    # 返回：权重数组、排序后的 Morton 码、排序索引（用于将属性映射到 Morton 序）
    return W, val, reord



# Morton 解码：将 Morton 码还原为 3D 坐标
def val2V(val, factor):
    '''

    Parameters
    ----------
    val : morton code
    factor : shift morton code for deocoding

    Returns
    -------
    V_re : point cloud

    '''

    if factor>2 or factor<0:
        print('error')
        return

    # 左移 factor 位，用于在不同深度层级的解码中补偿移位
    val = val<<factor
    V_re = np.zeros((val.shape[0],3))

    # 解码 x 坐标：从 Morton 码中每隔 3 位提取一位（位 0,3,6,...）
    V_re[:,2] = (0x000001 & val) + \
                (0x000002 & (val>> 2)) + \
                (0x000004 & (val>> 4)) + \
                (0x000008 & (val>> 6)) + \
                (0x000010 & (val>> 8)) + \
                (0x000020 & (val>>10)) + \
                (0x000040 & (val>>12)) + \
                (0x000080 & (val>>14)) + \
                (0x000100 & (val>>16)) + \
                (0x000200 & (val>>18)) + \
                (0x000400 & (val>>20)) + \
                (0x000800 & (val>>22)) + \
                (0x001000 & (val>>24)) + \
                (0x002000 & (val>>26)) + \
                (0x004000 & (val>>28)) + \
                (0x008000 & (val>>30)) + \
                (0x010000 & (val>>32)) + \
                (0x020000 & (val>>34)) + \
                (0x040000 & (val>>36)) + \
                (0x080000 & (val>>38)) + \
                (0x100000 & (val>>40))
    # 被注释掉的更高位解码


    # 解码 y 坐标：从 Morton 码中每隔 3 位提取一位（位 1,4,7,...）
    V_re[:,1] = (0x000001 & (val>> 1)) + \
                (0x000002 & (val>> 3)) + \
                (0x000004 & (val>> 5)) + \
                (0x000008 & (val>> 7)) + \
                (0x000010 & (val>> 9)) + \
                (0x000020 & (val>>11)) + \
                (0x000040 & (val>>13)) + \
                (0x000080 & (val>>15)) + \
                (0x000100 & (val>>17)) + \
                (0x000200 & (val>>19)) + \
                (0x000400 & (val>>21)) + \
                (0x000800 & (val>>23)) + \
                (0x001000 & (val>>25)) + \
                (0x002000 & (val>>27)) + \
                (0x004000 & (val>>29)) + \
                (0x008000 & (val>>31)) + \
                (0x010000 & (val>>33)) + \
                (0x020000 & (val>>35)) + \
                (0x040000 & (val>>37)) + \
                (0x080000 & (val>>39)) + \
                (0x100000 & (val>>41))
    # 被注释掉的更高位解码


    # 解码 z 坐标：从 Morton 码中每隔 3 位提取一位（位 2,5,8,...）
    V_re[:,0] = (0x000001 & (val>> 2)) + \
                (0x000002 & (val>> 4)) + \
                (0x000004 & (val>> 6)) + \
                (0x000008 & (val>> 8)) + \
                (0x000010 & (val>>10)) + \
                (0x000020 & (val>>12)) + \
                (0x000040 & (val>>14)) + \
                (0x000080 & (val>>16)) + \
                (0x000100 & (val>>18)) + \
                (0x000200 & (val>>20)) + \
                (0x000400 & (val>>22)) + \
                (0x000800 & (val>>24)) + \
                (0x001000 & (val>>26)) + \
                (0x002000 & (val>>28)) + \
                (0x004000 & (val>>30)) + \
                (0x008000 & (val>>32)) + \
                (0x010000 & (val>>34)) + \
                (0x020000 & (val>>36)) + \
                (0x040000 & (val>>38)) + \
                (0x080000 & (val>>40)) + \
                (0x100000 & (val>>42))
    # + \
    #             (0x200000 & (val>>44)) + \
    #             (0x400000 & (val>>46)) + \
    #             (0x800000 & (val>>48))

    # 根据 factor 补偿坐标偏移（对应 Morton 码在不同轴方向的移位解码）
    if factor == 1:
        V_re[:,2]/=2
    if factor == 2:
        V_re[:,1]/=2
        V_re[:,2]/=2


    return V_re


# ============================================================
# Haar 小波变换矩阵（正向/逆向），用于 RAHT 中相邻节点属性的变换
# 正向变换：将两个节点的属性转换为低频/高频系数
# 矩阵形式：[[sqrt(w0), sqrt(w1)], [-sqrt(w1), sqrt(w0)]] / sqrt(w0+w1)
# ============================================================

def transform_batched(a0, a1, C0, C1):
    # input
    # a0, a1: float, weight（归一化后的权重 sqrt(w_i)/sqrt(w0+w1)）
    # C0, C1: np.array (n,), att of vertices（左右邻居节点属性）

    # output
    # v0, v1: np.array (n,), trans att of vertices（变换后的低频/高频系数）

    # 构造 2x2 Haar 变换矩阵（批量化）
    trans_matrix=np.array([[a0, a1],
                           [-a1, a0]])
    trans_matrix=trans_matrix.transpose((2,0,1))


    V=np.matmul(trans_matrix, np.concatenate((C0,C1),1))

    return V[:,0], V[:,1]

def transform_batched_torch(a0, a1, C0, C1):
    # transform_batched 的 PyTorch GPU 加速版本
    # 直接用逐元素运算代替矩阵乘法，效率更高
    t0 = torch.tensor(a0[:,None]).cuda().float()
    t1 = torch.tensor(a1[:,None]).cuda().float()
    # V0 = a0*C0 + a1*C1（低频系数）
    V0 = t0*C0+t1*C1
    # V1 = -a1*C0 + a0*C1（高频系数）
    V1 = -t1*C0+t0*C1

    # temp1 = a0[:,None]
    # temp2 = a1[:,None]
    # trans_matrix = np.concatenate((temp1, temp2, -temp2, temp1),1)
    # trans_matrix = trans_matrix.reshape(-1,2,2)
    # trans_matrix = torch.tensor(trans_matrix).to(C0.get_device()).float()

    # print('trans_matrix.shape', trans_matrix.shape)
    # print('trans_matrix.shape', C0.shape)
    # print('torch.cat((C0,C1),1).shape', torch.cat((C0,C1),1).shape)
    # V=torch.matmul(trans_matrix, torch.cat((C0,C1),1))

    return V0, V1


# 逆 Haar 小波变换（NumPy 版本）：将低频/高频系数还原为原始属性
def itransform_batched(a0, a1, CT0, CT1):
    # input
    # a0, a1: float, weight
    # CT0, CT1: np.array (n,), trans att of vertices

    # output
    # c0, c1: np.array (n,), att of vertices

    # 逆变换矩阵：正变换的转置（因为正交矩阵的逆等于转置）
    trans_matrix=np.array([[a0, -a1],
                           [a1, a0]])
    trans_matrix=trans_matrix.transpose((2,0,1))

    C=np.matmul(trans_matrix, np.concatenate((CT0,CT1),1))

    return C[:,0], C[:,1]


def itransform_batched_torch(a0, a1, CT0, CT1):
    # 逆变换的 PyTorch GPU 加速版本
    # CT0 = 低频系数，CT1 = 高频系数
    # c0 = a0*CT0 - a1*CT1, c1 = a1*CT0 + a0*CT1

    t0 = torch.tensor(a0[:,None]).cuda().float()
    t1 = torch.tensor(a1[:,None]).cuda().float()
    V0 = t0*CT0-t1*CT1
    V1 = t1*CT0+t0*CT1

    return V0, V1


# ============================================================
# RAHT (Region-Adaptive Hierarchical Transform) 核心函数
# ============================================================
# RAHT 是一种基于八叉树结构的三维点云属性变换，类似于 3D Haar 小波变换。
# 核心思想：沿 Morton 码排序后的点云，逐层对相邻节点进行 Haar 小波变换，
# 生成低频系数（近似）和高频系数（细节），实现属性的层级化表示。
#
# 流程概述：
# 1. 将点云坐标编码为 Morton 码并排序
# 2. 逐深度层遍历（每层对应 x/y/z 一个轴方向）
# 3. 根据 Morton 码判断相邻节点是否属于同一父节点
# 4. 对同一父节点下的两个子节点执行 Haar 小波变换
# 5. 收集低频/高频系数及辅助信息
# ============================================================


def haar3D(inV, inC, depth):
    '''


    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    inC : attributes
    depth : depth level of geometry(octree)

    Returns
    -------
    res : transformed coefficients and side information

    """
    RAHT 正向变换（NumPy 版本）—— 完整版，收集所有辅助信息

    '''


    import copy
    inC = copy.deepcopy(inC)


    # N = 点数，NN = 初始点数（用于后续索引计算），K = 属性维度
    N, K = inC.shape
    NN = N

    # 八叉树深度 × 3 = RAHT 变换的总层数（每层八叉树沿 x,y,z 各变换一次）
    depth *= 3
    # print('depth', depth)

    # low_freq coeffs for transmitting coeffs (high_freq)
    # low_freq = np.zeros(inC.shape)





    # 预分配下一层的权重、Morton 码、位置映射
    wT = np.zeros((N, )).astype(np.uint64)
    valT = np.zeros((N, )).astype(np.uint64)
    posT = np.zeros((N, )).astype(np.int64)


    # 高频系数对应的 3D 坐标（-1 表示未赋值）
    node_xyz = np.zeros((N, 3))-1

    # 每个系数对应的变换深度（-1 表示未赋值）
    depth_CT = np.zeros((N, ))-1





    # Morton 编码：计算权重、Morton 码、排序映射
    w, val, TMP = copyAsort(inV)



    # pos: 当前层的位置映射（初始为顺序索引）
    # C: 按 Morton 序排列的属性
    pos = np.arange(N)
    C = inC[TMP].astype(np.float64)



    # 用于存储每层辅助信息的列表
    iCT_low=[]   # 每层的低频系数
    iparent=[]   # 每层的父节点索引
    iW=[]        # 每层的权重
    iPos=[]      # 每层的节点坐标




    for d in range(depth):
        # S = 当前层的节点数
        S = N

        # ---------- 判断相邻节点是否属于同一父节点 ----------
        # 将 Morton 码最低位清零后比较：若相邻两个 Morton 码仅最低位不同，
        # 说明它们在当前轴方向上相邻，属于同一父节点，可以合并
        # 例：Morton 码 2(10) 和 3(11) → 清零最低位后均为 2(10) → 可合并
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE
        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))

        # 当前层的两类节点索引：
        # comb_idx_array: 可合并的左邻居索引（mask=True 的位置）
        # trans_idx_array: 直接传递（不参与合并）的节点索引
        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)

        # ---------- 下一层的索引和掩码 ----------
        # idxT_array: 去掉右邻居后的索引（左邻居+传递节点），即下一层的节点
        # maskT: 对应这些节点在下一层是合并(True)还是传递(False)
        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array]


        # ---------- 计算下一层的权重 ----------
        # 不合并的节点：权重直接传递
        # 合并的节点：权重相加（w_left + w_right）
        wT[np.where(maskT==False)[0]] = w[trans_idx_array]
        wT[np.where(maskT==True)[0]] = w[comb_idx_array]+w[comb_idx_array+1]



        # ---------- 位置映射更新 ----------
        # pos: 当前层属性 C 与 Morton 序的映射
        # posT: 下一层属性 C 与 Morton 序的映射
        posT[np.where(maskT==False)[0]] = pos[trans_idx_array]
        posT[np.where(maskT==True)[0]] = pos[comb_idx_array]



        # ---------- Haar 小波变换 ----------
        # 对可合并的相邻节点对执行属性变换
        # left_node_array = 左邻居索引，right_node_array = 右邻居索引
        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1
        # a = sqrt(w_left + w_right)，用于归一化
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))
        C[pos[left_node_array]], C[pos[right_node_array]] = transform_batched(np.sqrt((w[left_node_array]))/a,
                                                  np.sqrt((w[right_node_array]))/a,
                                                  C[pos[left_node_array],None],
                                                  C[pos[right_node_array],None])




        # ---------- 收集辅助信息 ----------
        # parent: 当前层每个节点对应的父节点索引（用于树结构追踪）
        parent=np.arange(S)
        parent_t=np.zeros(S)
        parent_t[right_node_array]=1
        parent_t = parent_t.cumsum()
        parent = parent-parent_t
        iparent.append(parent.astype(int))


        # 收集下一层的低频系数、权重、坐标
        iCT_low.append(C[pos[idxT_array]])

        num_nodes = N-comb_idx_array.shape[0]
        iW.append(wT[:num_nodes]+0)

        # 计算下一层节点的 3D 坐标（从 Morton 码解码）
        # d%3 决定当前层沿哪个轴方向变换：0→x, 1→y, 2→z
        Pos_t = val2V(val, d%3)[idxT_array]
        if d%3 == 0:
            Pos_t[:,2]=Pos_t[:,2]//2
        if d%3 == 1:
            Pos_t[:,1]=Pos_t[:,1]//2
        if d%3 == 2:
            Pos_t[:,0]=Pos_t[:,0]//2
        iPos.append(Pos_t)


        # 记录高频系数对应的 3D 坐标（右邻居节点的坐标）
        node_xyz[pos[right_node_array]] = val2V(val[right_node_array], d%3)

        if d%3 == 0:
            node_xyz[pos[right_node_array],2]=node_xyz[pos[right_node_array],2]//2
        if d%3 == 1:
            node_xyz[pos[right_node_array],1]=node_xyz[pos[right_node_array],1]//2
        if d%3 == 2:
            node_xyz[pos[right_node_array],0]=node_xyz[pos[right_node_array],0]//2


        # 记录每个系数的变换深度
        depth_CT[pos[trans_idx_array]] = d
        depth_CT[pos[left_node_array]], depth_CT[pos[right_node_array]] = d, d

        # ---------- 辅助信息收集完毕 ----------


        # ---------- 为下一层做准备 ----------
        # valT: 下一层的 Morton 码（右移 1 位，相当于八叉树上移一层）
        valT = (val >> 1)[idxT_array]

        # N_T = 当前节点数，N = 下一层节点数 = 当前节点数 - 合并对数
        N_T=N
        N=N-comb_idx_array.shape[0]


        # 将高频系数的位置和权重移到数组末尾
        # 前半部分存储下一层的低频系数，后半部分存储高频系数
        N_idx_array=np.arange(N_T, N, -1)-NN-1
        wT[N_idx_array]=wT[np.where(maskT==True)[0]]
        posT[N_idx_array]=pos[comb_idx_array+1]

        # 更新 pos 和 w 的后半部分为高频系数的信息
        pos[N:S] = posT[N:S]
        w[N:S] = wT[N:S]

        # 交换当前层和下一层的变量，为下一次迭代做准备
        val, valT = valT, val
        pos, posT = posT, pos
        w, wT = wT, w

    # 输出最终权重（按 pos 映射回原始顺序）
    outW=np.zeros(w.shape)
    outW[pos]=w


    res = {'CT':C,           # 变换后的系数（含低频和高频）
           'w':outW,         # 最终权重
           'depth_CT':depth_CT, # 每个系数的变换深度
           'node_xyz':node_xyz, # 高频系数对应的 3D 坐标
        #    'low_freq':low_freq,

           'iCT_low':iCT_low,   # 各层低频系数
           'iW':iW,             # 各层权重
           'iPos':iPos,         # 各层节点坐标

           'iparent':iparent,   # 各层父节点索引
           }

    return res

def haar3D_torch(inC, depth, w, val, TMP):
    # haar3D 的 PyTorch GPU 加速版本
    # 与 haar3D 逻辑相同，但：
    # 1. 属性变换使用 GPU 加速的 transform_batched_torch
    # 2. 不收集辅助信息（iCT_low, iW, iPos, iparent），仅返回变换后系数
    # 3. Morton 编码结果 (w, val, TMP) 由外部传入，避免重复计算
    '''


    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    inC : attributes
    depth : depth level of geometry(octree)

    Returns
    -------
    res : transformed coefficients and side information

    '''

    N, K = inC.shape
    NN = N

    # 八叉树深度 × 3 = RAHT 变换总层数
    depth *= 3

    wT = np.zeros((N, )).astype(np.uint64)
    valT = np.zeros((N, )).astype(np.uint64)
    posT = np.zeros((N, )).astype(np.int64)

    # 高频系数对应的 3D 坐标
    node_xyz = np.zeros((N, 3))-1

    # Morton 编码结果由外部传入，这里直接使用
    pos = np.arange(N)
    # 按 Morton 序排列属性（GPU 张量）
    C = inC[torch.tensor(TMP)]




    for d in range(depth):
        S = N
        # 判断相邻节点是否属于同一父节点（与 haar3D 逻辑相同）
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE
        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))

        # 合并索引和传递索引
        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)

        # 下一层的索引和掩码
        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array]

        # 计算下一层权重
        wT[np.where(maskT==False)[0]] = w[trans_idx_array]
        wT[np.where(maskT==True)[0]] = w[comb_idx_array]+w[comb_idx_array+1]

        # 位置映射更新
        posT[np.where(maskT==False)[0]] = pos[trans_idx_array]
        posT[np.where(maskT==True)[0]] = pos[comb_idx_array]

        # GPU 加速的 Haar 小波变换
        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))
        C[pos[left_node_array]], C[pos[right_node_array]] = transform_batched_torch(np.sqrt((w[left_node_array]))/a,
                                                  np.sqrt((w[right_node_array]))/a,
                                                  C[pos[left_node_array],None],
                                                  C[pos[right_node_array],None])

        # 辅助信息收集（大部分被注释掉，仅保留 node_xyz 计算）
        parent=np.arange(S)
        parent_t=np.zeros(S)
        parent_t[right_node_array]=1
        parent_t = parent_t.cumsum()
        parent = parent-parent_t


        # 计算下一层节点的 3D 坐标（从 Morton 码解码）
        Pos_t = val2V(val, d%3)[idxT_array]
        if d%3 == 0:
            Pos_t[:,2]=Pos_t[:,2]//2
        if d%3 == 1:
            Pos_t[:,1]=Pos_t[:,1]//2
        if d%3 == 2:
            Pos_t[:,0]=Pos_t[:,0]//2

        # 记录高频系数对应的 3D 坐标
        node_xyz[pos[right_node_array]] = val2V(val[right_node_array], d%3)

        if d%3 == 0:
            node_xyz[pos[right_node_array],2]=node_xyz[pos[right_node_array],2]//2
        if d%3 == 1:
            node_xyz[pos[right_node_array],1]=node_xyz[pos[right_node_array],1]//2
        if d%3 == 2:
            node_xyz[pos[right_node_array],0]=node_xyz[pos[right_node_array],0]//2

        # ---------- 为下一层做准备 ----------
        # valT: 下一层的 Morton 码（右移 1 位，相当于八叉树上移一层）
        valT = (val >> 1)[idxT_array]

        # N_T = 当前节点数，N = 下一层节点数 = 当前节点数 - 合并对数
        N_T=N
        N=N-comb_idx_array.shape[0]

        # 将高频系数的位置和权重移到数组末尾
        N_idx_array=np.arange(N_T, N, -1)-NN-1
        wT[N_idx_array]=wT[np.where(maskT==True)[0]]
        posT[N_idx_array]=pos[comb_idx_array+1]

        # 更新 pos 和 w 的后半部分为高频系数的信息
        pos[N:S] = posT[N:S]
        w[N:S] = wT[N:S]

        # 交换当前层和下一层的变量，为下一次迭代做准备
        val, valT = valT, val
        pos, posT = posT, pos
        w, wT = wT, w

    return C



def get_RAHT_tree(inV, depth):
    # 构建 RAHT 树结构（不执行属性变换）
    # 仅记录每层的 Morton 码、权重、节点数，用于后续解码
    '''


    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    depth : depth level of geometry(octree)

    Returns
    -------
    res : tree without low- and high-freq coeffs

    """
    从底向上构建 RAHT 树，记录各层的 Morton 码 (iVAL)、权重 (iW) 和节点数 (iM)

    '''


    # N = 点数，NN = 初始点数
    N, _ = inV.shape
    NN = N

    # 八叉树深度 × 3 = RAHT 变换总层数
    depth *= 3

    wT = np.zeros((N, ))
    valT = np.zeros((N, )).astype(np.uint64)
    posT = np.zeros((N, )).astype(np.uint64)

    # 每层的 Morton 码和权重（预分配最大空间）
    iVAL = np.zeros((depth, N)).astype(np.uint64)
    iW = np.zeros((depth, N))

    # M = 当前层节点数，初始等于总点数
    M = N
    # 每层的节点数
    iM = np.zeros((depth, )).astype(np.uint64)

    # Morton 编码
    w, val, reord = copyAsort(inV)
    pos = np.arange(N).astype(np.uint64)


    # 从底向上构建 RAHT 树，记录各层信息用于解码
    for d in range(depth):

        # 记录当前层的 Morton 码、权重、节点数
        iVAL[d,:M] = val[:M]
        iW[d,:M] = w[:M]
        iM[d]= M

        M = 0
        S = N

        # 判断相邻节点是否可合并（与编码过程相同）
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE
        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))

        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)

        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array]

        # 更新下一层的权重和位置
        wT[np.where(maskT==False)[0]] = w[trans_idx_array]
        posT[np.where(maskT==False)[0]] = pos[trans_idx_array]

        wT[np.where(maskT==True)[0]] = w[comb_idx_array]+w[comb_idx_array+1]
        posT[np.where(maskT==True)[0]] = pos[comb_idx_array]

        # 下一层的 Morton 码（右移 1 位）
        valT = (val >> 1)[idxT_array]

        # 更新节点数
        N_T=N
        N=N-comb_idx_array.shape[0]
        M=N

        # 将合并节点的信息移到数组末尾
        N_idx_array=np.arange(N_T, N, -1)-NN-1
        wT[N_idx_array]=wT[np.where(maskT==True)[0]]
        posT[N_idx_array]=pos[comb_idx_array+1]

        # 更新后半部分
        pos[N:S] = posT[N:S]
        w[N:S] = wT[N:S]

        # 交换变量
        val, valT = valT, val
        pos, posT = posT, pos
        w, wT = wT, w

    # 返回树结构信息
    # reord: Morton 排序索引（将原始属性映射到 Morton 序）
    # pos: 最终的位置映射（将 Morton 序映射到系数数组索引）
    # iVAL, iW, iM: 各层的 Morton 码、权重、节点数
    res = {'reord':reord,
           'pos':pos,
           'iVAL':iVAL,
           'iW':iW,
           'iM':iM,
           }

    return res






def inv_haar3D(inV, inCT, depth):
    # RAHT 逆变换（NumPy 版本）
    # 从变换系数恢复原始属性
    '''


    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    inCT : transformed coeffs (high-freq coeffs)
    depth : depth level of geometry(octree)

    Returns
    -------
    res : rec attributes

    """
    从顶向下逐层执行逆 Haar 小波变换，将低频/高频系数还原为原始属性

    '''

    # N = 点数，K = 属性维度
    N, K = inCT.shape
    NN = N

    # 八叉树深度 × 3 = RAHT 变换总层数
    depth *= 3

    CT = np.zeros((N, K))
    C = np.zeros((N, K))
    outC = np.zeros((N, K))

    # 构建或获取 RAHT 树结构
    res_tree = get_RAHT_tree(inV, depth)
    reord, pos, iVAL, iW, iM = \
        res_tree['reord'], res_tree['pos'], res_tree['iVAL'], res_tree['iW'], res_tree['iM']

    # 按 pos 映射排列系数
    CT = inCT[pos]
    C = np.zeros(CT.shape)

    # 从顶向下逐层解码
    d = depth

    while d:

        d = d-1
        # S = 当前层的节点数
        S = iM[d]
        M = iM[d-1] if d else NN

        # 获取当前层的 Morton 码和权重
        val, w = iVAL[d, :int(S)], iW[d, :int(S)]

        M = 0
        N = S

        # 判断相邻节点是否可合并（与编码过程相同的索引计算）
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE

        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))

        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)

        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array.astype(int)]

        # 直接传递的低频系数（不参与合并的节点）
        C[trans_idx_array] = CT[np.where(maskT==False)[0]]

        # 对合并的节点对执行逆 Haar 小波变换
        # 将低频和高频系数还原为两个原始属性

        # N_idx_array: 高频系数在数组末尾的索引
        N_T=N
        N=N-comb_idx_array.shape[0]
        N_idx_array=np.arange(N_T, N, -1)-NN-1

        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1

        # 逆变换：c0 = a0*CT0 - a1*CT1, c1 = a1*CT0 + a0*CT1
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))
        C[left_node_array], C[right_node_array] = itransform_batched(np.sqrt((w[left_node_array]))/a,
                                        np.sqrt((w[right_node_array]))/a,
                                        CT[np.where(maskT==True)[0]][:,None],
                                        CT[N_idx_array.astype(int)][:,None])

        # 更新系数数组，为下一层解码做准备
        CT[:S] = C[:S]

    # 按 reord 映射回原始属性顺序
    outC[reord] = C

    return outC

def inv_haar3D_torch(inCT, depth, res_tree):
    # RAHT 逆变换（PyTorch GPU 加速版本）
    # res_tree 由外部预计算并传入，避免重复构建树结构
    '''
    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    inCT : transformed coeffs (high-freq coeffs)
    depth : depth level of geometry(octree)

    Returns
    -------
    res : rec attributes

    """
    GPU 加速的逆变换，逻辑与 inv_haar3D 相同，但使用 torch 张量运算

    '''

    # N = 点数，K = 属性维度
    N, K = inCT.shape
    NN = N

    # 八叉树深度 × 3 = RAHT 变换总层数
    depth *= 3

    outC = torch.zeros((N, K), device='cuda')

    reord, pos, iVAL, iW, iM = \
        res_tree['reord'], res_tree['pos'], res_tree['iVAL'], res_tree['iW'], res_tree['iM']

    # 按 pos 映射排列系数（GPU 张量）
    CT = inCT[torch.tensor(pos.astype(np.int64))]
    C = torch.zeros(CT.shape, device='cuda')

    # 从顶向下逐层解码
    d = depth

    while d:

        d = d-1
        # S = 当前层的节点数
        S = iM[d]
        M = iM[d-1] if d else NN

        # 获取当前层的 Morton 码和权重
        val, w = iVAL[d, :int(S)], iW[d, :int(S)]

        M = 0
        N = S

        # 判断相邻节点是否可合并（与编码过程相同的索引计算）
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE

        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))

        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)

        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array.astype(int)]

        # 直接传递的低频系数
        C[trans_idx_array] = CT[np.where(maskT==False)[0]]

        # 对合并的节点对执行 GPU 加速的逆 Haar 小波变换

        # N_idx_array: 高频系数在数组末尾的索引
        N_T=N
        N=N-comb_idx_array.shape[0]
        N_idx_array=np.arange(N_T, N, -1)-NN-1

        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1
        # 逆变换：c0 = a0*CT0 - a1*CT1, c1 = a1*CT0 + a0*CT1
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))
        C[left_node_array], C[right_node_array] = itransform_batched_torch(np.sqrt((w[left_node_array]))/a,
                                        np.sqrt((w[right_node_array]))/a,
                                        CT[np.where(maskT==True)[0]][:,None],
                                        CT[N_idx_array.astype(int)][:,None])

        # 更新系数数组，为下一层解码做准备
        CT[:S] = C[:S]

    # 按 reord 映射回原始属性顺序
    outC[reord] = C
    return outC

def haar3D_param(depth, w, val):
    # 预计算 RAHT 正向变换所需的参数（权重和索引）
    # 不执行实际变换，仅记录变换参数，用于后续高效的正向变换
    """
    Parameters
    ----------
    depth : depth level of geometry(octree)
    w : weight array (from copyAsort)
    val : morton code array (from copyAsort)

    Returns
    -------
    res : 预计算的变换参数（权重和索引）

    """
    N = val.shape[0]
    NN = N
    depth *= 3

    wT = np.zeros((N, )).astype(np.uint64)
    valT = np.zeros((N, )).astype(np.uint64)
    posT = np.zeros((N, )).astype(np.int64)

    pos = np.arange(N)

    # 各层预计算的变换参数
    iW1 = []          # 左邻居归一化权重 sqrt(w_left)/sqrt(w_left+w_right)
    iW2 = []          # 右邻居归一化权重 sqrt(w_right)/sqrt(w_left+w_right)
    iLeft_idx = []    # 左邻居在属性数组中的位置索引
    iRight_idx = []   # 右邻居在属性数组中的位置索引
    iPos = []

    for d in range(depth):
        S = N

        # 判断相邻节点是否可合并（与其他函数相同）
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE
        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))

        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)

        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array]

        wT[np.where(maskT==False)[0]] = w[trans_idx_array]
        wT[np.where(maskT==True)[0]] = w[comb_idx_array]+w[comb_idx_array+1]

        posT[np.where(maskT==False)[0]] = pos[trans_idx_array]
        posT[np.where(maskT==True)[0]] = pos[comb_idx_array]

        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))

        # 记录变换参数
        iW1.append(np.sqrt((w[left_node_array]))/a)
        iW2.append(np.sqrt((w[right_node_array]))/a)

        iLeft_idx.append(pos[left_node_array]+0)
        iRight_idx.append(pos[right_node_array]+0)

        # 下一层的 Morton 码
        valT = (val >> 1)[idxT_array]

        N_T=N
        N=N-comb_idx_array.shape[0]

        # 将合并节点的信息移到数组末尾
        N_idx_array=np.arange(N_T, N, -1)-NN-1
        wT[N_idx_array]=wT[np.where(maskT==True)[0]]
        posT[N_idx_array]=pos[comb_idx_array+1]

        pos[N:S] = posT[N:S]
        w[N:S] = wT[N:S]

        # 交换变量
        val, valT = valT, val
        pos, posT = posT, pos
        w, wT = wT, w

    outW=np.zeros(w.shape)
    outW[pos]=w

    res = {
        'w':outW,           # 最终权重
        'iW1':iW1,          # 各层左邻居归一化权重
        'iW2':iW2,          # 各层右邻居归一化权重
        'iLeft_idx':iLeft_idx,    # 各层左邻居位置索引
        'iRight_idx':iRight_idx,  # 各层右邻居位置索引
        }

    return res


def inv_haar3D_param(inV, depth):
    # 预计算 RAHT 逆变换所需的参数（权重和索引）
    # 先构建 RAHT 树，再从顶向下计算各层的逆变换参数
    """
    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    depth : depth level of geometry(octree)

    Returns
    -------
    res : 预计算的逆变换参数（权重、索引）

    """
    N = inV.shape[0]
    NN = N
    depth *= 3

    # 构建 RAHT 树
    res_tree = get_RAHT_tree(inV, depth)
    reord, pos, iVAL, iW, iM = \
        res_tree['reord'], res_tree['pos'], res_tree['iVAL'], res_tree['iW'], res_tree['iM']

    # 各层预计算的逆变换参数
    iW1 = []              # 左邻居归一化权重
    iW2 = []              # 右邻居归一化权重
    iS = []               # 各层节点数
    iLeft_idx = []        # 左邻居在当前层系数数组中的索引
    iRight_idx = []       # 右邻居在当前层系数数组中的索引

    iLeft_idx_CT = []     # 左邻居在上一层系数数组中的索引（低频系数位置）
    iRight_idx_CT = []    # 右邻居在上一层系数数组中的索引（高频系数位置）

    iTrans_idx = []       # 直接传递节点的索引
    iTrans_idx_CT = []    # 直接传递节点在上一层系数数组中的索引

    # 从顶向下逐层计算逆变换参数
    d = depth

    while d:
        d = d-1
        S = iM[d]
        M = iM[d-1] if d else NN

        val, w = iVAL[d, :int(S)], iW[d, :int(S)]
        M = 0
        N = S

        # 判断相邻节点是否可合并（与其他函数相同）
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE

        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))

        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)

        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array.astype(int)]

        N_T=N
        N=N-comb_idx_array.shape[0]
        N_idx_array=np.arange(N_T, N, -1)-NN-1

        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))

        # 记录逆变换参数
        iW1.append(np.sqrt((w[left_node_array]))/a)
        iW2.append(np.sqrt((w[right_node_array]))/a)

        iLeft_idx.append(left_node_array.astype(int)+0)
        iRight_idx.append(right_node_array.astype(int)+0)

        iLeft_idx_CT.append(np.where(maskT==True)[0].astype(int))
        iRight_idx_CT.append(N_idx_array.astype(int))

        iTrans_idx.append(trans_idx_array)
        iTrans_idx_CT.append(np.where(maskT==False)[0])

        iS.append(S)

    res = {
        'pos': pos,             # 最终位置映射
        'iS':iS,                # 各层节点数

        'iW1':iW1,              # 各层左邻居归一化权重
        'iW2':iW2,              # 各层右邻居归一化权重
        'iLeft_idx':iLeft_idx,        # 各层左邻居索引
        'iRight_idx':iRight_idx,      # 各层右邻居索引

        'iLeft_idx_CT':iLeft_idx_CT,    # 各层左邻居在上一层系数中的索引
        'iRight_idx_CT':iRight_idx_CT,  # 各层右邻居在上一层系数中的索引

        'iTrans_idx':iTrans_idx,        # 各层传递节点索引
        'iTrans_idx_CT':iTrans_idx_CT,  # 各层传递节点在上一层系数中的索引
        }

    return res


def haar3D_gpu(xyz, attributes, depth):
    """
    PyTorch GPU 版 RAHT 正向变换（一行调用）

    Parameters
    ----------
    xyz : np.ndarray (N, 3)
        体素化后的点云坐标（整数）
    attributes : np.ndarray (N, K) 或 torch.Tensor (N, K)
        点云属性（如 RGB 颜色）
    depth : int
        八叉树深度

    Returns
    -------
    CT : torch.Tensor (N, K), GPU float32
        变换系数，CT[0] 为 DC 分量，CT[1:] 为高频系数
    """
    w, val, reorder = copyAsort(xyz)
    res_fwd = haar3D_param(depth, w.copy(), val.copy())

    if isinstance(attributes, np.ndarray):
        C = torch.tensor(attributes[reorder], device='cuda', dtype=torch.float32)
    else:
        C = attributes[reorder].cuda().float()

    for d in range(depth * 3):
        C[res_fwd['iLeft_idx'][d]], C[res_fwd['iRight_idx'][d]] = transform_batched_torch(
            res_fwd['iW1'][d], res_fwd['iW2'][d],
            C[res_fwd['iLeft_idx'][d]], C[res_fwd['iRight_idx'][d]]
        )

    return C


def inv_haar3D_gpu(xyz, CT, depth):
    """
    PyTorch GPU 版 RAHT 逆向变换（一行调用）

    Parameters
    ----------
    xyz : np.ndarray (N, 3)
        体素化后的点云坐标（整数）
    CT : torch.Tensor (N, K), GPU
        正向变换输出的系数
    depth : int
        八叉树深度

    Returns
    -------
    rec : torch.Tensor (N, K), GPU float32
        恢复的属性，顺序与输入 haar3D_gpu 的 attributes 一致
    """
    w, val, reorder = copyAsort(xyz)
    res_inv = inv_haar3D_param(xyz, depth)

    # 按 pos 重排系数
    CT_in = CT[res_inv['pos']]
    OC = torch.zeros_like(CT_in)
    CT_temp = CT_in.clone()

    for i in range(depth * 3):
        S = res_inv['iS'][i]
        trans_idx = res_inv['iTrans_idx'][i]
        trans_CT = res_inv['iTrans_idx_CT'][i]
        left_idx = res_inv['iLeft_idx'][i]
        right_idx = res_inv['iRight_idx'][i]
        left_CT = res_inv['iLeft_idx_CT'][i]
        right_CT = res_inv['iRight_idx_CT'][i]

        OC[trans_idx] = CT_temp[trans_CT]
        OC[left_idx], OC[right_idx] = itransform_batched_torch(
            res_inv['iW1'][i], res_inv['iW2'][i],
            CT_temp[left_CT], CT_temp[right_CT]
        )
        CT_temp[:S] = OC[:S]

    rec = torch.zeros_like(OC)
    rec[reorder] = OC
    return rec
