import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from models.embedder import get_embedder

# from scipy.spatial import KDTree
# from torch.utils.data.sampler import WeightedRandomSampler
# from torch.distributions.categorical import Categorical
# from torch.distributions.uniform import Uniform
# 


def batched_index_select(values, indices, dim = 1):
  value_dims = values.shape[(dim + 1):]
  values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
  indices = indices[(..., *((None,) * len(value_dims)))]
  indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
  value_expand_len = len(indices_shape) - (dim + 1)
  values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

  value_expand_shape = [-1] * len(values.shape)
  expand_slice = slice(dim, (dim + value_expand_len))
  value_expand_shape[expand_slice] = indices.shape[expand_slice]
  values = values.expand(*value_expand_shape)

  dim += value_expand_len
  return values.gather(dim, indices)

def update_quaternion(delta_angle, prev_quat):
    s1 = 0
    s2 = prev_quat[0]
    v2 = prev_quat[1:]
    v1 = delta_angle / 2
    new_v = s1 * v2 + s2 * v1 + torch.cross(v1, v2)
    new_s = s1 * s2 - torch.sum(v1 * v2)
    new_quat = torch.cat([new_s.unsqueeze(0), new_v], dim=0)
    return new_quat

# def euler_to_quaternion(yaw, pitch, roll):
def euler_to_quaternion(roll, pitch, yaw):
    qx = torch.sin(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) - torch.cos(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)
    qy = torch.cos(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2)
    qz = torch.cos(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2) - torch.sin(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2)
    qw = torch.cos(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)
    
    # qx = torch.sin()
    return [qw, qx, qy, qz]
    # return [qx, qy, qz, qw]
    
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1) # -1 for the quaternion matrix # 
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    
    return o.reshape(quaternions.shape[:-1] + (3, 3))

