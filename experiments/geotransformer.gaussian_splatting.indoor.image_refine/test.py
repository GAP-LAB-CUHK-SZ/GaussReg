import argparse

import torch
import numpy as np

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error_w_scale

from config import make_cfg
from model import create_model
from plyfile import PlyData, PlyElement
import os.path as osp
from scipy.spatial.transform import Rotation
import random
import open3d as o3d
from geotransformer.utils.open3d import registration_with_ransac_from_correspondences
import pickle
import os
import time
from tqdm import tqdm
import json
from PIL import Image
import cv2

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', help="device")
    parser.add_argument("--coarse_reg_path", default='scannet_test_coarse', help="coarse registration output file")
    parser.add_argument("--scannet_path", default='ScanNet-GSReg', help="src point cloud numpy file")
    parser.add_argument("--fine_reg_path", default='scannet_test_fine', help="the path of saved fine registration output file")
    parser.add_argument("--weights", default='weights/fine_registration.pth.tar', help="model weights file")
    return parser

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def homo_warping(src_proj, ref_proj, depth_values, h, w):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, num_depth, height, width = depth_values.shape

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_values.device),
                               torch.arange(0, width, dtype=torch.float32, device=depth_values.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        mask = (proj_xy[:, 0, :, :] >= 0) * (proj_xy[:, 0, :, :] < w) * (proj_xy[:, 1, :, :] >= 0) * (proj_xy[:, 1, :, :] < h)
        score = mask.sum((-1,-2)) / (h * w)

    return score

def _read_ply_by_opacity(input_path, point_limit):
    """extract point cloud from gaussian splatting file"""
    plydata = PlyData.read(input_path)
    opacity = np.asarray(plydata.elements[0].data['opacity'])
    opacity = 1 / (1 + np.exp(-opacity))
    # index = np.where(opacity>0.7)[0]
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    index_x = (x < np.percentile(x, 90)) * (x > np.percentile(x, 10))
    index_y = (y < np.percentile(y, 90)) * (y > np.percentile(y, 10))
    index_z = (z < np.percentile(z, 90)) * (z > np.percentile(z, 10))
    index = np.where((opacity>0.7) * index_x * index_y * index_z)[0]
    if point_limit is not None and index.shape[0] > point_limit:
        indices = np.random.permutation(index.shape[0])[: point_limit]
        index = index[indices]
    points = np.stack([x,y,z], axis=1)
    features_dc = np.zeros((points.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
    features_extra = np.zeros((points.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))
    features = np.concatenate([features_dc, features_extra], axis=2)[index] #(N, 3, 16)
    points = points[index]
    center_point = points.mean(0)
    center_point[1] = center_point[1] - 2 * (points[:,1].max() - points[:,1].min())
    dir_pp = (points - center_point[None,:].repeat(points.shape[0], 0))
    dir_pp_normalized = dir_pp/(np.linalg.norm(dir_pp, axis=1, keepdims=True) + 1e-6)
    
    sh2rgb = eval_sh(3, features, dir_pp_normalized)
    colors = np.clip(sh2rgb + 0.5, 0.0, 1.0) * 255
    point_features = np.concatenate([opacity[index].reshape(points.shape[0], -1), colors.astype(np.float32)], axis=1)
    
    return points, point_features

def _load_point_cloud_from_ply(file_name, point_limit=None):
    points, point_features = _read_ply_by_opacity(file_name, point_limit)
    # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
    # if point_limit is not None and points.shape[0] > point_limit:
    #     indices = np.random.permutation(points.shape[0])[: point_limit]
    #     points = points[indices]
    #     point_features = point_features[indices]
    return points, point_features

def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation

def _augment_point_cloud(ref_points, src_points, rotation, translation):
    r"""Augment point clouds.

    ref_points = src_points @ rotation.T + translation

    1. Random rotation to one point cloud.
    2. Random noise.
    """
    aug_rotation = random_sample_rotation()
    scale = random.random() * 4 + 1
    if random.random() > 0.5:
        if random.random() > 0.5:
            aug_scale = scale
            src_points = src_points * aug_scale
            rotation = rotation / aug_scale 
        else:
            aug_scale = 1 / scale
            src_points = src_points * aug_scale
            rotation = rotation / aug_scale
    else:
        if random.random() > 0.5:
            aug_scale = scale
            ref_points = ref_points * aug_scale
            rotation = rotation * aug_scale 
            translation = translation * aug_scale
        else:
            aug_scale = 1 / scale
            ref_points = ref_points * aug_scale
            rotation = rotation * aug_scale
            translation = translation * aug_scale
    if random.random() > 0.5:
        ref_points = np.matmul(ref_points, aug_rotation.T)
        rotation = np.matmul(aug_rotation, rotation)
        translation = np.matmul(aug_rotation, translation)
    else:
        src_points = np.matmul(src_points, aug_rotation.T)
        rotation = np.matmul(rotation, aug_rotation.T)

    # ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
    # src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

    return ref_points, src_points, rotation, translation

def _adjust_point_cloud(ref_points, src_points, rotation, translation):
    r"""Adjust point clouds.

    ref_points = src_points @ rotation.T + translation

    """
    ref_volume = (ref_points[:,0].max() - ref_points[:,0].min()) * (ref_points[:,1].max() - ref_points[:,1].min()) * (ref_points[:,2].max() - ref_points[:,2].min())
    ref_adjust_scale = (10 / ref_volume) ** (1/3)
    ref_points = ref_points * ref_adjust_scale
    rotation = rotation * ref_adjust_scale
    translation = translation * ref_adjust_scale
    src_volume = (src_points[:,0].max() - src_points[:,0].min()) * (src_points[:,1].max() - src_points[:,1].min()) * (src_points[:,2].max() - src_points[:,2].min())
    src_adjust_scale = (10 / src_volume) ** (1/3)
    src_points = src_points * src_adjust_scale
    rotation = rotation / src_adjust_scale
    
    return ref_points, src_points, rotation, translation

def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

def _prepare_data(scene_path, new_dict_1, camera_1_intrinsics, camera_1_extrinsics, depth_1, A_ids):
    A_imgs = []
    A_intrinsics = []
    A_extrinsics = []
    for i,image_name in enumerate(A_ids):
        image_path = osp.join(scene_path, 'images', str(image_name)+'.jpg')
        # scale 0~255 to 0~1
        np_img_hr = np.array(Image.open(image_path), dtype=np.float32) / 255.
        height_hr, width_hr, _ = np_img_hr.shape
        height, width = height_hr // 64 * 32, width_hr // 64 * 32
        if i == 0:
            A_np_depth = depth_1[new_dict_1[str(image_name)]]
        np_img = cv2.resize(np_img_hr, (width, height), interpolation=cv2.INTER_NEAREST)
        A_imgs.append(np_img)
        intrinsics = np.eye(4)
        intrinsics[:3,:3] = camera_1_intrinsics[new_dict_1[str(image_name)]]
        intrinsics[0,:] = intrinsics[0,:] * width / width_hr
        intrinsics[1,:] = intrinsics[1,:] * height / height_hr
        A_intrinsics.append(intrinsics)
        extrinsics = camera_1_extrinsics[new_dict_1[str(image_name)]]
        A_extrinsics.append(extrinsics)
    A_imgs = np.stack(A_imgs).transpose([0, 3, 1, 2])
    A_intrinsics = np.stack(A_intrinsics)
    A_extrinsics = np.stack(A_extrinsics)
    A_near, A_far = np.percentile(A_np_depth, 5), np.percentile(A_np_depth, 95)
    center = (A_near + A_far) / 2
    distance = A_far - A_near
    A_near_new = center - 1.1 * distance / 2
    A_far_new = center + 1.5 * distance / 2
    A_depth_range = [[np.clip(A_near_new, 0, center)], [A_far_new]]
    return A_imgs, A_intrinsics, A_extrinsics, A_depth_range

def load_data(ref_file, src_file):
    ref_points, ref_feats = _load_point_cloud_from_ply(ref_file, 30000)
    src_points, src_feats = _load_point_cloud_from_ply(src_file, 30000)
    rotation = np.eye(3)
    translation = np.zeros(3)
    ref_points, src_points, rotation, translation = _augment_point_cloud(
                ref_points, src_points, rotation, translation
            )
    ref_volume = (ref_points[:,0].max() - ref_points[:,0].min()) * (ref_points[:,1].max() - ref_points[:,1].min()) * (ref_points[:,2].max() - ref_points[:,2].min())
    ref_adjust_scale = 1.
    src_adjust_scale = 1.
    if ref_volume > 50:
        ref_adjust_scale = (50 / ref_volume) ** (1/3)
        ref_points = ref_points * ref_adjust_scale
        rotation = rotation * ref_adjust_scale
        translation = translation * ref_adjust_scale
    elif ref_volume < 10:
        ref_adjust_scale = (10 / ref_volume) ** (1/3)
        ref_points = ref_points * ref_adjust_scale
        rotation = rotation * ref_adjust_scale
        translation = translation * ref_adjust_scale
    src_volume = (src_points[:,0].max() - src_points[:,0].min()) * (src_points[:,1].max() - src_points[:,1].min()) * (src_points[:,2].max() - src_points[:,2].min())
    if src_volume > 50:
        src_adjust_scale = (50 / src_volume) ** (1/3)
        src_points = src_points * src_adjust_scale
        rotation = rotation / src_adjust_scale
    elif src_volume < 10:
        src_adjust_scale = (10 / src_volume) ** (1/3)
        src_points = src_points * src_adjust_scale
        rotation = rotation / src_adjust_scale
    # ref_points, src_points, rotation, translation = _adjust_point_cloud(
    #             ref_points, src_points, rotation, translation
    #         )
    transform = get_transform_from_rotation_translation(rotation, translation)
    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "ref_adjust_scale": ref_adjust_scale,
        "src_adjust_scale": src_adjust_scale,
    }
    data_dict["transform"] = transform.astype(np.float32)

    return data_dict

def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()
    device = args.device
    coarse_output_path = args.coarse_reg_path
    fine_output_path = args.fine_reg_path
    num_sample = 30
    estimated_transform_path = os.path.join(coarse_output_path, "estimated_transform.npz")

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])
    scannet_path = os.path.join(args.scannet_path, 'test')
    fine_time = []
    rre_list = []
    rte_list = []
    rse_list = []
    estimated_transform_list = {}
    gt_transform_list = {}
    estimated_transform_list = (np.load(estimated_transform_path, allow_pickle=True)["estimated_transform_list"]).item()
    scene_list = np.load(osp.join(args.scannet_path, f'test_transformations.npz'), allow_pickle=True)['transformations'].item()
    gt_transform_list = scene_list['gt_transformations_list']
    ref_transform_list = scene_list['ref_transformations_list']
    src_transform_list = scene_list['src_transformations_list']
    for scene_name in tqdm(gt_transform_list):
        scene_path = os.path.join(scannet_path, scene_name)
        estimated_transform = torch.from_numpy(estimated_transform_list[scene_name]).to(torch.float32).to(device)
        gt_transform = torch.from_numpy(gt_transform_list[scene_name]).to(torch.float32).to(device)
        ref_transform = torch.from_numpy(ref_transform_list[scene_name]).to(torch.float32).to(device)
        src_transform = torch.from_numpy(src_transform_list[scene_name]).to(torch.float32).to(device)
        ply1_camera_path = os.path.join(scene_path, 'A/output/cameras.json') 
        ply2_camera_path = os.path.join(scene_path, 'B/output/cameras.json') 
        with open(ply1_camera_path, "r") as file_1:
            camera_1_file = json.load(file_1) #w2c
        with open(ply2_camera_path, "r") as file_2:
            camera_2_file = json.load(file_2) #w2c
        ply1_depth_path = os.path.join(scene_path, 'A/output/train/ours_10000/render_depths/depths.npy') 
        ply2_depth_path = os.path.join(scene_path, 'B/output/train/ours_10000/render_depths/depths.npy') 
        depth_1 = torch.from_numpy(np.load(ply1_depth_path)).to(device) #(N,H,W)
        depth_2 = torch.from_numpy(np.load(ply2_depth_path)).to(device) #(M,H,W)
        camera_1_file_raw = camera_1_file.copy()
        camera_2_file_raw = camera_2_file.copy()
        depth_1_raw = depth_1.clone()
        depth_2_raw = depth_2.clone()
        depth_1 = depth_1[::round(len(depth_1) / num_sample)]
        depth_2 = depth_2[::round(len(depth_2) / num_sample)]
        camera_1_extrinsics = torch.zeros((len(camera_1_file), 4, 4)).to(device)
        camera_1_intrinsics = torch.zeros((len(camera_1_file), 3, 3)).to(device)
        for i in range(len(camera_1_file)):
            camera_1_extrinsics[i,:3,:3] = torch.tensor(camera_1_file[i]['rotation']).to(device)
            camera_1_extrinsics[i,:3, 3] = torch.tensor(camera_1_file[i]['position']).to(device)
            camera_1_extrinsics[i, 3, 3] = 1.
            camera_1_extrinsics[i] = torch.inverse(camera_1_extrinsics[i])
            camera_1_intrinsics[i, 0, 0] = camera_1_file[i]['fx'] * depth_1[0].shape[1] / camera_1_file[i]['width']
            camera_1_intrinsics[i, 1, 1] = camera_1_file[i]['fy'] * depth_1[0].shape[0] / camera_1_file[i]['height']
            camera_1_intrinsics[i, 0, 2] = depth_1[0].shape[1] / 2
            camera_1_intrinsics[i, 1, 2] = depth_1[0].shape[0] / 2
            camera_1_intrinsics[i, 2, 2] = 1.
        camera_1_extrinsics = torch.matmul(camera_1_extrinsics, torch.inverse(ref_transform)[None,:,:]) #(M,4,4)
        camera_1_extrinsics_raw = camera_1_extrinsics.clone()
        camera_1_intrinsics_raw = camera_1_intrinsics.clone()
        camera_1_extrinsics = camera_1_extrinsics[::round(len(camera_1_extrinsics) / num_sample)]
        camera_1_intrinsics = camera_1_intrinsics[::round(len(camera_1_intrinsics) / num_sample)]
        camera_2_extrinsics = torch.zeros((len(camera_2_file), 4, 4)).to(device)
        camera_2_intrinsics = torch.zeros((len(camera_2_file), 3, 3)).to(device)
        for i in range(len(camera_2_file)):
            camera_2_extrinsics[i,:3,:3] = torch.tensor(camera_2_file[i]['rotation']).to(device)
            camera_2_extrinsics[i,:3, 3] = torch.tensor(camera_2_file[i]['position']).to(device)
            camera_2_extrinsics[i, 3, 3] = 1.
            camera_2_extrinsics[i] = torch.inverse(camera_2_extrinsics[i])
            camera_2_intrinsics[i, 0, 0] = camera_2_file[i]['fx'] * depth_2[0].shape[1] / camera_2_file[i]['width']
            camera_2_intrinsics[i, 1, 1] = camera_2_file[i]['fy'] * depth_2[0].shape[0] / camera_2_file[i]['height']
            camera_2_intrinsics[i, 0, 2] = depth_2[0].shape[1] / 2
            camera_2_intrinsics[i, 1, 2] = depth_2[0].shape[0] / 2
            camera_2_intrinsics[i, 2, 2] = 1.
        camera_2_extrinsics = torch.matmul(camera_2_extrinsics, torch.inverse(src_transform)[None,:,:]) #(M,4,4)
        camera_2_extrinsics = torch.matmul(camera_2_extrinsics, torch.inverse(estimated_transform)[None,:,:]) #(M,4,4)
        camera_2_extrinsics_raw = camera_2_extrinsics.clone()
        camera_2_intrinsics_raw = camera_2_intrinsics.clone()
        camera_2_extrinsics = camera_2_extrinsics[::round(len(camera_2_extrinsics) / num_sample)]
        camera_2_intrinsics = camera_2_intrinsics[::round(len(camera_2_intrinsics) / num_sample)]
        camera_1_new = camera_1_extrinsics.clone()
        camera_1_new[:, :3, :4] = torch.matmul(camera_1_intrinsics, camera_1_extrinsics[:, :3, :4]) #(N,4,4)
        camera_2_new = camera_2_extrinsics.clone()
        camera_2_new[:, :3, :4] = torch.matmul(camera_2_intrinsics, camera_2_extrinsics[:, :3, :4]) #(M,4,4)
        camera_1_direction = torch.matmul(camera_1_extrinsics[:, :3, :3].permute(0,2,1), torch.tensor([0.,0.,1.])[None,:,None].to(device))[:,:,0] #(N,3)
        camera_2_direction = torch.matmul(camera_2_extrinsics[:, :3, :3].permute(0,2,1), torch.tensor([0.,0.,1.])[None,:,None].to(device))[:,:,0] #(M,3)
        camera_2_direction = camera_2_direction / (camera_2_direction[0]**2).sum()
        
        camera_1_1_direction_score = torch.matmul(camera_1_direction, camera_1_direction.permute(1,0)) #(N,N)
        camera_1_2_direction_score = torch.matmul(camera_1_direction, camera_2_direction.permute(1,0)) #(N,M)
        camera_2_2_direction_score = torch.matmul(camera_2_direction, camera_2_direction.permute(1,0)) #(M,M)
        camera_1_1_direction_values, camera_1_1_direction_indices = camera_1_1_direction_score.topk(camera_1_1_direction_score.shape[1] // 3, dim=1, largest=True, sorted=True)
        camera_1_2_direction_values, camera_1_2_direction_indices = camera_1_2_direction_score.topk(camera_1_2_direction_score.shape[1] // 3, dim=1, largest=True, sorted=True)
        camera_2_2_direction_values, camera_2_2_direction_indices = camera_2_2_direction_score.topk(camera_2_2_direction_score.shape[1] // 3, dim=1, largest=True, sorted=True)
        camera_1_1_new = torch.gather(camera_1_new[None, :].repeat(camera_1_new.shape[0], 1, 1, 1), 1, camera_1_1_direction_indices[:,:,None,None].repeat(1,1,4,4)) #(N,10,4,4)
        camera_1_2_new = torch.gather(camera_2_new[None, :].repeat(camera_1_new.shape[0], 1, 1, 1), 1, camera_1_2_direction_indices[:,:,None,None].repeat(1,1,4,4)) #(N,10,4,4)
        camera_2_2_new = torch.gather(camera_2_new[None, :].repeat(camera_2_new.shape[0], 1, 1, 1), 1, camera_2_2_direction_indices[:,:,None,None].repeat(1,1,4,4)) #(M,10,4,4)
        depth_1_1 = torch.gather(depth_1[None, :].repeat(depth_1.shape[0], 1, 1, 1), 1, camera_1_1_direction_indices[:,:,None,None].repeat(1,1,depth_1.shape[1],depth_1.shape[2])) #(N,10,4,4)
        depth_1_2 = torch.gather(depth_2[None, :].repeat(depth_1.shape[0], 1, 1, 1), 1, camera_1_2_direction_indices[:,:,None,None].repeat(1,1,depth_2.shape[1],depth_2.shape[2])) #(N,10,4,4)
        depth_2_2 = torch.gather(depth_2[None, :].repeat(depth_2.shape[0], 1, 1, 1), 1, camera_2_2_direction_indices[:,:,None,None].repeat(1,1,depth_2.shape[1],depth_2.shape[2])) #(N,10,4,4)
        score_1_1_1 = homo_warping(camera_1_new[:, None, :,:].repeat(1,camera_1_1_new.shape[1], 1, 1).view(-1,4,4), camera_1_1_new.view(-1,4,4), depth_1_1.view(-1, 1, depth_1.shape[1], depth_1.shape[2]), depth_1.shape[1], depth_1.shape[2])
        score_1_1_2 = homo_warping(camera_1_1_new.view(-1,4,4), camera_1_new[:, None, :,:].repeat(1,camera_1_1_new.shape[1], 1, 1).view(-1,4,4), depth_1[:,None,:,:].repeat(1,camera_1_1_new.shape[1], 1, 1).view(-1,1,depth_1.shape[1], depth_1.shape[2]), depth_1.shape[1], depth_1.shape[2])
        score_1_2_1 = homo_warping(camera_1_new[:, None, :,:].repeat(1,camera_1_2_new.shape[1], 1, 1).view(-1,4,4), camera_1_2_new.view(-1,4,4), depth_1_2.view(-1, 1, depth_2.shape[1], depth_2.shape[2]), depth_1.shape[1], depth_1.shape[2])
        score_1_2_2 = homo_warping(camera_1_2_new.view(-1,4,4), camera_1_new[:, None, :,:].repeat(1,camera_1_2_new.shape[1], 1, 1).view(-1,4,4), depth_1[:,None,:,:].repeat(1,camera_1_2_new.shape[1], 1, 1).view(-1,1,depth_1.shape[1], depth_1.shape[2]), depth_2.shape[1], depth_2.shape[2])
        score_2_2_1 = homo_warping(camera_2_new[:, None, :,:].repeat(1,camera_2_2_new.shape[1], 1, 1).view(-1,4,4), camera_2_2_new.view(-1,4,4), depth_2_2.view(-1, 1, depth_2.shape[1], depth_2.shape[2]), depth_2.shape[1], depth_2.shape[2])
        score_2_2_2 = homo_warping(camera_2_2_new.view(-1,4,4), camera_2_new[:, None, :,:].repeat(1,camera_2_2_new.shape[1], 1, 1).view(-1,4,4), depth_2[:,None,:,:].repeat(1,camera_2_2_new.shape[1], 1, 1).view(-1,1,depth_2.shape[1], depth_2.shape[2]), depth_2.shape[1], depth_2.shape[2])

        score_1_1 = (score_1_1_2 + score_1_1_1).view_as(camera_1_1_direction_indices) / 2 + camera_1_1_direction_values
        score_1_2 = (score_1_2_2 + score_1_2_1).view_as(camera_1_2_direction_indices) / 2 + camera_1_2_direction_values 
        score_2_2 = (score_2_2_2 + score_2_2_1).view_as(camera_2_2_direction_indices) / 2 + camera_2_2_direction_values
        score = score_1_2.max(-1)
        _, indices = score[0].topk(5, 0, largest=True, sorted=True)
        sub_indices = score[1][indices]
        sub_indices = torch.cat([indices.unsqueeze(1), sub_indices.unsqueeze(1)], dim=1)
        camera_1_file = camera_1_file[::round(len(camera_1_file) / num_sample)]
        camera_2_file = camera_2_file[::round(len(camera_2_file) / num_sample)]
        A_id = int(camera_1_file[sub_indices[0,0]]['img_name'].split('_')[-1])
        _, A_indices = score_1_1[sub_indices[0,0]].topk(5, 0, largest=True, sorted=True)
        A_ids = [A_id] + [int(camera_1_file[camera_1_1_direction_indices[sub_indices[0,0], A_indices[i + 1]]]['img_name'].split('_')[-1]) for i in range(4)]
        B_id = int(camera_2_file[camera_1_2_direction_indices[sub_indices[0,0], sub_indices[0,1]]]['img_name'].split('_')[-1])
        _, B_indices = score_2_2[camera_1_2_direction_indices[sub_indices[0,0], sub_indices[0,1]]].topk(5, 0, largest=True, sorted=True)
        B_ids = [B_id] + [int(camera_2_file[camera_2_2_direction_indices[camera_1_2_direction_indices[sub_indices[0,0], sub_indices[0,1]], B_indices[i + 1]]]['img_name'].split('_')[-1]) for i in range(4)]

        new_dict_1 = {k['img_name'] : i for i, k in enumerate(camera_1_file_raw)}
        new_dict_2 = {k['img_name'] : i for i, k in enumerate(camera_2_file_raw)}
        A_imgs, A_intrinsics, A_extrinsics, A_depth_range = _prepare_data(os.path.join(scene_path, 'A'), new_dict_1, camera_1_intrinsics_raw.cpu(), camera_1_extrinsics_raw.cpu(), depth_1_raw.cpu(), A_ids)
        B_imgs, B_intrinsics, B_extrinsics, B_depth_range = _prepare_data(os.path.join(scene_path, 'B'), new_dict_2, camera_2_intrinsics_raw.cpu(), camera_2_extrinsics_raw.cpu(), depth_2_raw.cpu(), B_ids)
    
        data_dict = {}
        data_dict['A_imgs'] = torch.from_numpy(A_imgs[None,...]).to(torch.float32)
        data_dict['A_intrinsics'] = torch.from_numpy(A_intrinsics[None,...]).to(torch.float32)
        data_dict['A_extrinsics'] = torch.from_numpy(A_extrinsics[None,...]).to(torch.float32)
        data_dict['A_depth_range'] = torch.tensor(A_depth_range).to(torch.float32)
        data_dict['B_imgs'] = torch.from_numpy(B_imgs[None,...])
        data_dict['B_intrinsics'] = torch.from_numpy(B_intrinsics[None,...]).to(torch.float32)
        data_dict['B_extrinsics'] = torch.from_numpy(B_extrinsics[None,...]).to(torch.float32)
        data_dict['B_depth_range'] = torch.tensor(B_depth_range).to(torch.float32)
        data_dict['ref_extrinsics'] = torch.from_numpy(A_extrinsics[0][None,...]).to(torch.float32)
        data_dict['src_extrinsics'] = torch.from_numpy(B_extrinsics[0][None,...]).to(torch.float32)
        data_dict['transform'] = gt_transform[None,...].to(torch.float32)
        # prediction
        data_dict = to_cuda(data_dict)
        start_time = time.time()
        model.eval()
        output_dict = model(data_dict)
        end_time = time.time()
        fine_time.append((end_time - start_time)*1000)
        data_dict = release_cuda(data_dict)
        output_dict = release_cuda(output_dict)

        # get results
        estimated_transform_new = output_dict["estimated_transform"]
        transform = data_dict["transform"][0]

        estimated_transform_final = estimated_transform_new @ estimated_transform.cpu().numpy()
        estimated_transform_list[str(scene_name)] = estimated_transform_final

        if not osp.exists(os.path.join(fine_output_path, str(scene_name))):
            os.makedirs(os.path.join(fine_output_path, str(scene_name)))
        ref_pcd = o3d.io.read_point_cloud(os.path.join(coarse_output_path, str(scene_name), 'ref.ply'))
        src_pcd = o3d.io.read_point_cloud(os.path.join(coarse_output_path, str(scene_name), 'src.ply'))
        src_pcd_pred = src_pcd.transform(estimated_transform_new)
        o3d.io.write_point_cloud(os.path.join(fine_output_path, str(scene_name), 'ref.ply'), ref_pcd)
        o3d.io.write_point_cloud(os.path.join(fine_output_path, str(scene_name), 'src.ply'), src_pcd_pred)
        src_pcd_gt = src_pcd.transform(transform@np.linalg.inv(estimated_transform.cpu().numpy()))
        o3d.io.write_point_cloud(os.path.join(fine_output_path, str(scene_name), 'ref_gt.ply'), ref_pcd)
        o3d.io.write_point_cloud(os.path.join(fine_output_path, str(scene_name), 'src_gt.ply'), src_pcd_gt)

        # compute error
        gt_transform_list[str(scene_name)] = transform
        rre, rte, rse = compute_registration_error_w_scale(transform, estimated_transform_final)
        rre_list.append(rre)
        rte_list.append(rte)
        rse_list.append(rse)
        
    print("fine_time:", np.array(fine_time).mean())
    print("rre_avg:", np.array(rre_list).mean())
    print("rte_avg:", np.array(rte_list).mean())
    print("rse_avg:", np.array(rse_list).mean())
    print("rre < 5:", (np.array(rre_list) < 5).sum()/np.array(rre_list).shape[0])
    print("rre < 10:", (np.array(rre_list) < 10).sum()/np.array(rre_list).shape[0])
    print("rte < 0.1:", (np.array(rte_list) < 0.1).sum()/np.array(rte_list).shape[0])
    print("rte < 0.2:", (np.array(rte_list) < 0.2).sum()/np.array(rte_list).shape[0])
    print("rse < 0.1:", (np.array(rse_list) < 0.1).sum()/np.array(rse_list).shape[0])
    print("rse < 0.2:", (np.array(rse_list) < 0.2).sum()/np.array(rse_list).shape[0])

if __name__ == "__main__":
    main()
