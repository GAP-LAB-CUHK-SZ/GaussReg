import numpy as np
from plyfile import PlyData, PlyElement
import torch
import torch.nn.functional as F
import os
import shutil
from geotransformer.utils.graphics_utils import *

def sh_values(sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2] (N,3,15)
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    N, C, _ = sh.shape
    sh = sh[:,:,None,:] #(N,3,1,15)
    # dirs = np.random.randn(15,3)
    # dirs = (dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8))[None, None, :, :] #(1,1,15,3)
    sh_values_1 = np.zeros((N,C,3,3)) #(N,C,3,3)
    x, y, z = dirs[..., 0:3, 0], dirs[..., 0:3, 1], dirs[..., 0:3, 2] #(1,1,3)
    sh_values_1[:,:,:,0] = - C1 * y
    sh_values_1[:,:,:,1] = C1 * z 
    sh_values_1[:,:,:,2] = - C1 * x
    sh_values_2 = np.zeros((N,C,5,5)) #(N,C,5,5)
    x, y, z = dirs[..., 3:8, 0], dirs[..., 3:8, 1], dirs[..., 3:8, 2] #(1,1,5)
    xx, yy, zz = x * x, y * y, z * z #(1,1,5)
    xy, yz, xz = x * y, y * z, x * z #(1,1,5)
    sh_values_2[:,:,:,0] = C2[0] * xy
    sh_values_2[:,:,:,1] = C2[1] * yz
    sh_values_2[:,:,:,2] = C2[2] * (2.0 * zz - xx - yy) 
    sh_values_2[:,:,:,3] = C2[3] * xz
    sh_values_2[:,:,:,4] = C2[4] * (xx - yy)
    sh_values_3 = np.zeros((N,C,7,7)) #(N,C,7,7)
    x, y, z = dirs[..., 8:15, 0], dirs[..., 8:15, 1], dirs[..., 8:15, 2] #(1,1,7)
    xx, yy, zz = x * x, y * y, z * z #(1,1,7)
    xy, yz, xz = x * y, y * z, x * z #(1,1,7)
    sh_values_3[:,:,:,0] = C3[0] * y * (3 * xx - yy)
    sh_values_3[:,:,:,1] = C3[1] * xy * z
    sh_values_3[:,:,:,2] = C3[2] * y * (4 * zz - xx - yy)
    sh_values_3[:,:,:,3] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
    sh_values_3[:,:,:,4] = C3[4] * x * (4 * zz - xx - yy)
    sh_values_3[:,:,:,5] = C3[5] * z * (xx - yy)
    sh_values_3[:,:,:,6] = C3[6] * x * (xx - 3 * yy)
    return sh_values_1, sh_values_2, sh_values_3

def sh_rotation(sh, sh_dc, rotation):
    # sh (N,3,15)
    # rotation (3, 3)
    dirs = np.random.randn(15,3)
    dirs = (dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)) #(15,3)
    sh_values_1, sh_values_2, sh_values_3 = sh_values(sh, dirs[None, None, :, :])
    dirs_rotation = dirs @ rotation.T #(15,3)
    sh_values_1_rotation, sh_values_2_rotation, sh_values_3_rotation = sh_values(sh, dirs_rotation[None, None, :, :])
    sh_transform_1 = np.linalg.pinv(sh_values_1) @ sh_values_1_rotation #(N,C,3,3)
    sh_transform_2 = np.linalg.pinv(sh_values_2) @ sh_values_2_rotation
    sh_transform_3 = np.linalg.pinv(sh_values_3) @ sh_values_3_rotation
    sh_rotation = np.zeros_like(sh)
    sh_rotation[:,:,0:3] = (sh[:,:,None,0:3] @ sh_transform_1)[:,:,0,:]
    sh_rotation[:,:,3:8] = (sh[:,:,None,3:8] @ sh_transform_2)[:,:,0,:]
    sh_rotation[:,:,8:15] = (sh[:,:,None,8:15] @ sh_transform_3)[:,:,0,:]
    return sh_rotation

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
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

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    for i in range(45):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l

def save_ply(xyz, f_dc, f_rest, opacities, scale, rotation, path):
    normals = np.zeros_like(xyz)
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def load_ply(path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots

def gaussian_fuse(input_path_1, input_path_2, transform_path, output_path):
    """fuse gaussian splatting file"""
    xyz_1, features_dc_1, features_extra_1, opacities_1, scales_1, rots_1 = load_ply(input_path_1)
    xyz_2, features_dc_2, features_extra_2, opacities_2, scales_2, rots_2 = load_ply(input_path_2)
    
    estimated_transform = np.load(transform_path)["estimated_transform"]
    rotation = estimated_transform[:3, :3]  # (3, 3)
    translation = estimated_transform[None, :3, 3]  # (1, 3)
    scale = (rotation @ rotation.T)[0,0] ** 0.5
    rotation = rotation / scale
    xyz_2_transform = xyz_2 @ rotation.T * scale + translation
    scales_2_transform = scales_2 + np.log(scale) if scale != 1. else scales_2
    rot_matrix_2_transform = torch.matmul(torch.tensor(rotation[None,:,:], dtype=torch.float), quaternion_to_matrix(torch.tensor(rots_2, dtype=torch.float)))
    rots_2_transform = matrix_to_quaternion(rot_matrix_2_transform).cpu().numpy()
    features_extra_2_transform = sh_rotation(features_extra_2, features_dc_2, rotation)
    
    # xyz_2_transform = xyz_2
    # scales_2_transform = scales_2
    # rots_2_transform = rots_2
    # features_extra_2_transform = features_extra_2

    xyz_1_center = xyz_1.mean(0)
    xyz_2_center = xyz_2_transform.mean(0)
    index_1 = np.linalg.norm(xyz_1 - xyz_1_center, axis=1) < np.linalg.norm(xyz_1 - xyz_2_center, axis=1)
    index_2 = np.linalg.norm(xyz_2_transform - xyz_2_center, axis=1) < np.linalg.norm(xyz_2_transform - xyz_1_center, axis=1)
    xyz = np.concatenate([xyz_1[index_1], xyz_2_transform[index_2]], axis=0)
    features_dc = np.concatenate([features_dc_1[index_1], features_dc_2[index_2]], axis=0)
    features_extra = np.concatenate([features_extra_1[index_1], features_extra_2_transform[index_2]], axis=0)
    opacities = np.concatenate([opacities_1[index_1], opacities_2[index_2]], axis=0)
    scales = np.concatenate([scales_1[index_1], scales_2_transform[index_2]], axis=0)
    rots = np.concatenate([rots_1[index_1], rots_2_transform[index_2]], axis=0)
    save_ply(xyz, features_dc.reshape(xyz.shape[0], -1), features_extra.reshape(xyz.shape[0], -1), opacities, scales, rots, output_path)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Fusion script parameters")
    parser.add_argument('--root_path', type=str, default="scene_name")
    parser.add_argument('--transform_path', type=str, default='demo_outputs/estimated_transform.npz')
    args, extras = parser.parse_known_args()
    root_path = args.root_path
    input_path_1 = os.path.join(root_path, "A/output/point_cloud/iteration_30000/point_cloud.ply")
    input_path_2 = os.path.join(root_path, "B/output/point_cloud/iteration_30000/point_cloud.ply")
    transform_path = args.transform_path
    os.makedirs(os.path.join(root_path, "fuse/output/point_cloud/iteration_30000"))
    shutil.copy(os.path.join(root_path, "A/output/cameras.json"), os.path.join(root_path, "fuse/output/cameras.json"))
    shutil.copy(os.path.join(root_path, "A/output/cfg_args"), os.path.join(root_path, "fuse/output/cfg_args"))
    output_path = os.path.join(root_path, "fuse/output/point_cloud/iteration_30000/point_cloud.ply")
    gaussian_fuse(input_path_1, input_path_2, transform_path, output_path)
