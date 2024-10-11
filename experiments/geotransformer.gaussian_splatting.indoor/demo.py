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
import os
import random
import open3d as o3d
from geotransformer.utils.graphics_utils import *
import fpsample

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default="scene_name/B/output/point_cloud/iteration_30000/point_cloud.ply", help="src point cloud numpy file")
    parser.add_argument("--ref_file", default="scene_name/A/output/point_cloud/iteration_30000/point_cloud.ply", help="ref point cloud numpy file")
    parser.add_argument("--output_path", default='demo_outputs', help="output file path")
    parser.add_argument("--weights", default='weights/coarse_registration.pth.tar', help="model weights file")
    parser.add_argument("--num_sample", type=int, default=30000, help="number of sample points")
    return parser

def _read_ply_by_opacity(input_path, point_limit):
    """extract point cloud from gaussian splatting file"""
    plydata = PlyData.read(input_path)
    opacity = np.asarray(plydata.elements[0].data['opacity'])
    opacity = 1 / (1 + np.exp(-opacity))
    # index = np.where(opacity>0.7)[0]
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    index_x = (x < np.percentile(x, 95)) * (x > np.percentile(x, 5))
    index_y = (y < np.percentile(y, 95)) * (y > np.percentile(y, 5))
    index_z = (z < np.percentile(z, 95)) * (z > np.percentile(z, 5))
    index = np.where((opacity>0.7) * index_x * index_y * index_z)[0]
    points = np.stack([x,y,z], axis=1)
    if point_limit is not None and index.shape[0] > point_limit:
        # indices = np.random.permutation(index.shape[0])[: point_limit]
        fps_samples_idx = fpsample.bucket_fps_kdline_sampling(points[index], point_limit, h=9)
        index = index[fps_samples_idx]
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
    max_length = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    center_point = center_point + np.array([0, 2 * max_length, 0])

    dir_pp = (points - center_point[None,:].repeat(points.shape[0], 0))
    dir_pp_normalized = dir_pp/(np.linalg.norm(dir_pp, axis=1, keepdims=True) + 1e-6)
    
    sh2rgb = eval_sh(3, features, dir_pp_normalized)
    colors = np.clip(sh2rgb + 0.5, 0.0, 1.0) * 255
    point_features = np.concatenate([opacity[index].reshape(points.shape[0], -1), colors.astype(np.float32)], axis=1)
    
    
    return points, point_features

def _load_point_cloud_from_ply(file_name, point_limit=None):
    points, point_features = _read_ply_by_opacity(file_name, point_limit)
    return points, point_features

def load_data(args):
    ref_points, ref_feats = _load_point_cloud_from_ply(args.ref_file, args.num_sample)
    src_points, src_feats = _load_point_cloud_from_ply(args.src_file, args.num_sample)

    ref_volume = (ref_points[:,0].max() - ref_points[:,0].min()) * (ref_points[:,1].max() - ref_points[:,1].min()) * (ref_points[:,2].max() - ref_points[:,2].min())
    ref_center = (ref_points.max(0) + ref_points.min(0)) / 2
    ref_points = ref_points - ref_center

    src_volume = (src_points[:,0].max() - src_points[:,0].min()) * (src_points[:,1].max() - src_points[:,1].min()) * (src_points[:,2].max() - src_points[:,2].min())
    src_center = (src_points.max(0) + src_points.min(0)) / 2
    src_points = src_points - src_center

    ref_adjust_scale = 1.
    src_adjust_scale = 1.

    if ref_volume > 50:
        ref_adjust_scale = (50 / ref_volume) ** (1/3)
        ref_points = ref_points * ref_adjust_scale

    elif ref_volume < 10:
        ref_adjust_scale = (30 / ref_volume) ** (1/3)
        ref_points = ref_points * ref_adjust_scale

    if src_volume > 50:
        src_adjust_scale = (50 / src_volume) ** (1/3)
        src_points = src_points * src_adjust_scale

    elif src_volume < 10:
        src_adjust_scale = (30 / src_volume) ** (1/3)
        src_points = src_points * src_adjust_scale

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "ref_adjust_scale": ref_adjust_scale,
        "src_adjust_scale": src_adjust_scale,
        "ref_center": ref_center,
        "src_center": src_center,

    }

    return data_dict

def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()

    # prepare data
    data_dict = load_data(args)
    ref_points_color = data_dict["ref_feats"][:,1:]
    src_points_color = data_dict["src_feats"][:,1:]
    neighbor_limits = [89, 30, 43, 49, 49] # default setting in GaussReg
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])
    model.eval()

    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    ref_points = output_dict["ref_points"].copy()
    src_points = output_dict["src_points"].copy()
    ref_adjust_scale = data_dict["ref_adjust_scale"]
    src_adjust_scale = data_dict["src_adjust_scale"]
    ref_center = data_dict["ref_center"].copy()
    src_center = data_dict["src_center"].copy()
    estimated_transform = output_dict["estimated_transform"].copy()

    ref_points_org = ref_points / ref_adjust_scale + ref_center
    src_points_org = src_points / src_adjust_scale + src_center
    ref_pcd_color = make_open3d_point_cloud(ref_points_org, ref_points_color / 255)
    ref_pcd_color.estimate_normals()
    src_pcd_color = make_open3d_point_cloud(src_points_org, src_points_color / 255)
    src_pcd_color.estimate_normals()

    if not osp.exists(args.output_path):
        os.makedirs(args.output_path)
    o3d.io.write_point_cloud(os.path.join(args.output_path, "point_cloud_src_org.ply"), src_pcd_color)

    estimated_transform_scale = np.zeros_like(estimated_transform)
    estimated_transform_scale[:3,:3] = estimated_transform[:3,:3] / ref_adjust_scale * src_adjust_scale
    estimated_transform_scale[:3, 3] = estimated_transform[:3, 3] / ref_adjust_scale + ref_center - np.matmul(estimated_transform_scale[:3,:3], src_center)
    estimated_transform_scale[3, 3] = 1.
    src_pcd_color = src_pcd_color.transform(estimated_transform_scale)
    o3d.io.write_point_cloud(os.path.join(args.output_path, "point_cloud_ref.ply"), ref_pcd_color)
    o3d.io.write_point_cloud(os.path.join(args.output_path, "point_cloud_src.ply"), src_pcd_color)
    np.savez(os.path.join(args.output_path, "estimated_transform.npz"), estimated_transform = estimated_transform_scale)

if __name__ == "__main__":
    main()
