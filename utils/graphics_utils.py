#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
from typing import NamedTuple

import numpy as np
import torch


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def matrix_to_quaternion(rotation_matrices):
    """
    Convert rotation matrices to quaternions.
    
    Args:
        rotation_matrices: [N, 3, 3] rotation matrices
        
    Returns:
        quaternions: [N, 4] quaternions in (w, x, y, z) format
    """
    batch_size = rotation_matrices.shape[0]
    quaternions = torch.zeros(batch_size, 4, device=rotation_matrices.device, dtype=rotation_matrices.dtype)
    
    # Extract rotation matrix elements
    m00 = rotation_matrices[:, 0, 0]
    m01 = rotation_matrices[:, 0, 1]
    m02 = rotation_matrices[:, 0, 2]
    m10 = rotation_matrices[:, 1, 0]
    m11 = rotation_matrices[:, 1, 1]
    m12 = rotation_matrices[:, 1, 2]
    m20 = rotation_matrices[:, 2, 0]
    m21 = rotation_matrices[:, 2, 1]
    m22 = rotation_matrices[:, 2, 2]
    
    # Compute trace
    trace = m00 + m11 + m22
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
    quaternions[mask1, 0] = 0.25 * s1
    quaternions[mask1, 1] = (m21[mask1] - m12[mask1]) / s1
    quaternions[mask1, 2] = (m02[mask1] - m20[mask1]) / s1
    quaternions[mask1, 3] = (m10[mask1] - m01[mask1]) / s1
    
    # Case 2: m00 > m11 and m00 > m22
    mask2 = (~mask1) & (m00 > m11) & (m00 > m22)
    s2 = torch.sqrt(1.0 + m00[mask2] - m11[mask2] - m22[mask2]) * 2  # s = 4 * x
    quaternions[mask2, 0] = (m21[mask2] - m12[mask2]) / s2
    quaternions[mask2, 1] = 0.25 * s2
    quaternions[mask2, 2] = (m01[mask2] + m10[mask2]) / s2
    quaternions[mask2, 3] = (m02[mask2] + m20[mask2]) / s2
    
    # Case 3: m11 > m22
    mask3 = (~mask1) & (~mask2) & (m11 > m22)
    s3 = torch.sqrt(1.0 + m11[mask3] - m00[mask3] - m22[mask3]) * 2  # s = 4 * y
    quaternions[mask3, 0] = (m02[mask3] - m20[mask3]) / s3
    quaternions[mask3, 1] = (m01[mask3] + m10[mask3]) / s3
    quaternions[mask3, 2] = 0.25 * s3
    quaternions[mask3, 3] = (m12[mask3] + m21[mask3]) / s3
    
    # Case 4: else
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(1.0 + m22[mask4] - m00[mask4] - m11[mask4]) * 2  # s = 4 * z
    quaternions[mask4, 0] = (m10[mask4] - m01[mask4]) / s4
    quaternions[mask4, 1] = (m02[mask4] + m20[mask4]) / s4
    quaternions[mask4, 2] = (m12[mask4] + m21[mask4]) / s4
    quaternions[mask4, 3] = 0.25 * s4
    
    # Normalize quaternions
    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
    
    return quaternions
