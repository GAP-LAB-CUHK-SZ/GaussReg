import os.path as osp
import pickle
import random
from typing import Dict
from plyfile import PlyData, PlyElement
import os
from PIL import Image
import cv2
import fpsample

import numpy as np
import torch
import torch.utils.data
from geotransformer.utils.graphics_utils import *

from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
)
from geotransformer.utils.registration import get_correspondences

class ScanNetGSRegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,
    ):
        super(ScanNetGSRegDataset, self).__init__()

        self.dataset_root = dataset_root
        self.metadata_root = self.dataset_root #osp.join(self.dataset_root, 'metadata')
        self.data_root = self.dataset_root #osp.join(self.dataset_root, 'data')

        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        if self.subset != 'train':
            transform_list = np.load(osp.join(self.metadata_root, f'{subset}_transformations.npz'), allow_pickle=True)['transformations'].item()
            self.ref_transformations_list = transform_list['ref_transformations_list']
            self.src_transformations_list = transform_list['src_transformations_list']
            self.gt_transformations_list = transform_list['gt_transformations_list']
            
        with open(osp.join(self.metadata_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)
            if self.overlap_threshold is not None:
                self.metadata_list = [x for x in self.metadata_list if x['overlap'] > self.overlap_threshold]
            if self.subset != 'train':
                self.metadata_list = [x for x in self.metadata_list if x['scene_name'] in self.gt_transformations_list]

    def __len__(self):
        return len(self.metadata_list)
    
    def _read_ply_by_opacity(self, input_path, transformation):
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
        if transformation is None:
            center_point = points.mean(0)
            # center_point[1] = center_point[1] - 2 * (points[:,1].max() - points[:,1].min())
            max_length = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
            center_point = center_point + np.matmul(np.array([0, 2 * max_length, 0]), random_sample_rotation(self.aug_rotation).T)
        else:
            points = np.matmul(points, transformation[:3,:3].T) + transformation[:3,3][None,:]
            center_point = points.mean(0)
            max_length = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
            center_point = center_point + np.array([0, 2 * max_length, 0])

        dir_pp = (points - center_point[None,:].repeat(points.shape[0], 0))
        dir_pp_normalized = dir_pp/(np.linalg.norm(dir_pp, axis=1, keepdims=True) + 1e-6)
        sh2rgb = eval_sh(3, features, dir_pp_normalized)
        colors = np.clip(sh2rgb + 0.5, 0.0, 1.0) * 255
        point_features = np.concatenate([opacity[index].reshape(points.shape[0], -1), colors.astype(np.float32)], axis=1)

        return points, point_features

    def _load_point_cloud_from_ply(self, file_name, transformation=None):
        points, point_features = self._read_ply_by_opacity(osp.join(self.data_root, file_name), transformation)
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            # indices = np.random.permutation(points.shape[0])[: self.point_limit]
            indices = fpsample.bucket_fps_kdline_sampling(points, self.point_limit, h=9)
            points = points[indices]
            point_features = point_features[indices]
        return points, point_features

    def _adjust_point_cloud(self, ref_points, src_points, rotation, translation, max_adjust_volume=50, min_adjust_volume=10, apply_translation=False):
        r"""Adjust point clouds.

        ref_points = src_points @ rotation.T + translation

        """
        ref_volume = (ref_points[:,0].max() - ref_points[:,0].min()) * (ref_points[:,1].max() - ref_points[:,1].min()) * (ref_points[:,2].max() - ref_points[:,2].min())
        src_volume = (src_points[:,0].max() - src_points[:,0].min()) * (src_points[:,1].max() - src_points[:,1].min()) * (src_points[:,2].max() - src_points[:,2].min())
        ref_adjust_scale = 1.
        src_adjust_scale = 1.
        ref_center = np.zeros((3))
        src_center = np.zeros((3))
        if apply_translation == True:
            ref_center = (ref_points.max(0) + ref_points.min(0)) / 2
            ref_points = ref_points - ref_center
            src_center = (src_points.max(0) + src_points.min(0)) / 2
            src_points = src_points - src_center
        if ref_volume > 50:
            ref_adjust_scale = (max_adjust_volume / ref_volume) ** (1/3)
            ref_points = ref_points * ref_adjust_scale
            rotation = rotation * ref_adjust_scale
            translation = translation * ref_adjust_scale
        elif ref_volume < 10:
            ref_adjust_scale = (min_adjust_volume / ref_volume) ** (1/3)
            ref_points = ref_points * ref_adjust_scale
            rotation = rotation * ref_adjust_scale
            translation = translation * ref_adjust_scale
        if src_volume > 50:
            src_adjust_scale = (max_adjust_volume / src_volume) ** (1/3)
            src_points = src_points * src_adjust_scale
            rotation = rotation / src_adjust_scale
        elif src_volume < 10:
            src_adjust_scale = (min_adjust_volume / src_volume) ** (1/3)
            src_points = src_points * src_adjust_scale
            rotation = rotation / src_adjust_scale
        
        return ref_points, src_points, rotation, translation, ref_adjust_scale, src_adjust_scale, ref_center, src_center

    def _augment_point_cloud(self, ref_points, src_points, rotation, translation):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        scale = random.random() * 3 + 1
        if random.random() > 0.5:
            if random.random() > 0.5:
                aug_scale = scale
                src_points = src_points * aug_scale
                rotation = rotation / aug_scale 
            else:
                aug_scale = 1 / scale
                src_points = src_points * aug_scale
                rotation = rotation / aug_scale
        if random.random() > 0.5:
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

        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['ref_frame'] = metadata['frag_id0']
        data_dict['src_frame'] = metadata['frag_id1']
        data_dict['overlap'] = metadata['overlap']

        # get transformation
        if self.subset == 'train':
            rotation = metadata['rotation']
            translation = metadata['translation']

            # get point cloud
            ref_points, ref_features = self._load_point_cloud_from_ply(metadata['pcd0'])
            src_points, src_features = self._load_point_cloud_from_ply(metadata['pcd1'])
        else:
            rotation = self.gt_transformations_list[metadata['scene_name']][:3,:3]
            translation = self.gt_transformations_list[metadata['scene_name']][:3, 3]
            # get point cloud
            ref_points, ref_features = self._load_point_cloud_from_ply(metadata['pcd0'], self.ref_transformations_list[metadata['scene_name']])
            src_points, src_features = self._load_point_cloud_from_ply(metadata['pcd1'], self.src_transformations_list[metadata['scene_name']])

        if self.use_augmentation:
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )
        
        ref_points, src_points, rotation, translation, ref_adjust_scale, src_adjust_scale, ref_center, src_center = self._adjust_point_cloud(
            ref_points, src_points, rotation, translation, 
            min_adjust_volume=10 if self.subset == 'train' else 30, 
            apply_translation=False if self.subset == 'train' else True
        )

        transform = get_transform_from_rotation_translation(rotation, translation)

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = ref_features
        data_dict['src_feats'] = src_features
        data_dict['transform'] = transform.astype(np.float32)
        data_dict['ref_adjust_scale'] = ref_adjust_scale
        data_dict['src_adjust_scale'] = src_adjust_scale
        data_dict['ref_center'] = ref_center.astype(np.float32)
        data_dict['src_center'] = src_center.astype(np.float32)
        return data_dict