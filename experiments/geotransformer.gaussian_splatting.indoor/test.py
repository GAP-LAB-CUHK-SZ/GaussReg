import argparse

import torch
import numpy as np

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud
from geotransformer.utils.registration import compute_registration_error_w_scale

from config import make_cfg
from model import create_model
from plyfile import PlyData, PlyElement
import os.path as osp
from scipy.spatial.transform import Rotation
import open3d as o3d
import os
from tqdm import tqdm
from geotransformer.utils.graphics_utils import *
import fpsample

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet_path", default="ScanNet-GSReg", help="dataset path")
    parser.add_argument("--output_path", default='scannet_test_final', help="output file path")
    parser.add_argument("--weights", default='weights/coarse_registration.pth.tar', help="model weights file")
    parser.add_argument("--num_sample", type=int, default=30000, help="number of sample points")
    return parser

def _read_ply_by_opacity(input_path, point_limit, org_transform):
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
    points = np.matmul(points, org_transform[:3,:3].T) + org_transform[:3,3][None,:]
    center_point = points.mean(0)
    max_length = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    center_point = center_point + np.array([0, 2 * max_length, 0])
    
    dir_pp = (points - center_point[None,:].repeat(points.shape[0], 0))
    dir_pp_normalized = dir_pp/(np.linalg.norm(dir_pp, axis=1, keepdims=True) + 1e-6)
    
    sh2rgb = eval_sh(3, features, dir_pp_normalized)
    colors = np.clip(sh2rgb + 0.5, 0.0, 1.0) * 255
    point_features = np.concatenate([opacity[index].reshape(points.shape[0], -1), colors.astype(np.float32)], axis=1)

    
    return points, point_features

def _load_point_cloud_from_ply(file_name, point_limit, org_transform):
    points, point_features = _read_ply_by_opacity(file_name, point_limit, org_transform)
    return points, point_features

def load_data(ref_file, src_file, num_sample, ref_transform, src_transform, gt_transform):
    ref_points, ref_feats = _load_point_cloud_from_ply(ref_file, num_sample, ref_transform)
    src_points, src_feats = _load_point_cloud_from_ply(src_file, num_sample, src_transform)
    
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
    data_dict["transform"] = gt_transform.astype(np.float32)
    data_dict["ref_transform"] = ref_transform.astype(np.float32)
    data_dict["src_transform"] = src_transform.astype(np.float32)

    return data_dict

def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()
    output_path = args.output_path
    neighbor_limits = [89, 30, 43, 49, 49] # default setting in GaussReg

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])
    model.eval()

    scannet_path = os.path.join(args.scannet_path, 'test')
    scene_list = np.load(osp.join(args.scannet_path, f'test_transformations.npz'), allow_pickle=True)['transformations'].item()
    rre_list = []
    rte_list = []
    rse_list = []
    estimated_transform_list = {}
    gt_transform_list = scene_list['gt_transformations_list']
    ref_transform_list = scene_list['ref_transformations_list']
    src_transform_list = scene_list['src_transformations_list']
    for scene_name in tqdm(gt_transform_list):
        scene_path = os.path.join(scannet_path, scene_name)
        ref_ply_path = os.path.join(scene_path, 'A/output/point_cloud/iteration_10000/point_cloud.ply')
        src_ply_path = os.path.join(scene_path, 'B/output/point_cloud/iteration_10000/point_cloud.ply')
        # prepare data
        data_dict = load_data(ref_ply_path, src_ply_path, args.num_sample, ref_transform_list[scene_name], src_transform_list[scene_name], gt_transform_list[scene_name])
        ref_points_color = data_dict["ref_feats"][:,1:]
        src_points_color = data_dict["src_feats"][:,1:]
        data_dict = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
        )
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
        estimated_transform = output_dict["estimated_transform"].copy()
        transform = data_dict["transform"].copy()
        ref_center = data_dict["ref_center"].copy()
        src_center = data_dict["src_center"].copy()

        ref_points_org = ref_points / ref_adjust_scale + ref_center
        src_points_org = src_points / src_adjust_scale + src_center
        ref_pcd_color = make_open3d_point_cloud(ref_points_org, ref_points_color / 255)
        ref_pcd_color.estimate_normals()
        src_pcd_color = make_open3d_point_cloud(src_points_org, src_points_color / 255)
        src_pcd_color.estimate_normals()

        
        estimated_transform_scale = np.zeros_like(estimated_transform)

        estimated_transform_scale[:3,:3] = estimated_transform[:3,:3] / ref_adjust_scale * src_adjust_scale
        estimated_transform_scale[:3, 3] = estimated_transform[:3, 3] / ref_adjust_scale + ref_center - np.matmul(estimated_transform_scale[:3,:3], src_center)
        estimated_transform_scale[3, 3] = 1.

        src_pcd_color = src_pcd_color.transform(estimated_transform_scale)
        if not osp.exists(os.path.join(output_path, str(scene_name))):
            os.makedirs(os.path.join(output_path, str(scene_name)))
        o3d.io.write_point_cloud(os.path.join(output_path, str(scene_name), 'ref.ply'), ref_pcd_color)
        o3d.io.write_point_cloud(os.path.join(output_path, str(scene_name), 'src.ply'), src_pcd_color)
        
        estimated_transform_list[str(scene_name)] = estimated_transform_scale      
        # compute error
        rre, rte, rse = compute_registration_error_w_scale(transform, estimated_transform_scale)
        rre_list.append(rre)
        rte_list.append(rte)
        rse_list.append(rse)
    
    np.savez(os.path.join(output_path, "estimated_transform.npz"), estimated_transform_list = estimated_transform_list)
    np.savez(os.path.join(output_path, "rre_list.npz"), rre_list = np.array(rre_list))
    np.savez(os.path.join(output_path, "rte_list.npz"), rte_list = np.array(rte_list))
    np.savez(os.path.join(output_path, "rse_list.npz"), rse_list = np.array(rse_list))
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
