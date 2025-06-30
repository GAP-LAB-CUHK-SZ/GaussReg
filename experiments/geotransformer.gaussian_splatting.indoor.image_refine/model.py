import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import pdb
import random

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)

from backbone import FeatureNet, CostRegNet, Conv2d
from geotransformer.utils.open3d import registration_with_ransac_from_correspondences

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj.cpu()).to(src_fea.device))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

def _projection(fea, proj, depth_values):
    # fea: [B, C, H, W]
    # proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]

    batch, channels = fea.shape[0], fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = depth_values.shape[2], depth_values.shape[3]

    with torch.no_grad():
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]

    return proj_xyz.view(batch, 3, height, width)

def depth_regression_top2(p, cost_feature, depth_values):
    """
    p:(B,D,H,W)
    cost_feature:(B,C,D,H,W)
    depth_values:(B,D,H,W)
    """
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    prob_volume_sum2 = 2 * F.avg_pool3d(F.pad(p.unsqueeze(1), pad=(0, 0, 0, 0, 0, 0)), (2, 1, 1), stride=1, padding=0).squeeze(1)
    index = torch.max(prob_volume_sum2, dim=1, keepdim=True)[1]
    index_0 = (index.type(torch.float) + 1).type(torch.long)
    index = torch.cat((index, index_0), dim=1)
    prob = torch.gather(p, dim=1, index=index.type(torch.long))#(B,2,H,W)
    prob_0 = torch.sum(prob, dim=1, keepdim=False)
    prob = prob / torch.sum(prob, dim=1, keepdim=True)
    depth = torch.gather(depth_values, dim=1, index=index.type(torch.long))#(B,2,H,W)
    feature = torch.gather(cost_feature, dim=2, index=index[:,None].repeat(1,cost_feature.shape[1],1,1,1).type(torch.long))#(B,2,H,W)
    depth = torch.sum(prob * depth, 1)
    feature = torch.sum(prob[:,None,...] * feature, 2)

    return depth, feature, prob_0
    
class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius
        self.num_depth = cfg.model.num_depth
        self.backbone = FeatureNet(base_channels=cfg.backbone.base_channels, num_stage=cfg.backbone.num_stages)
        self.cost_regularization = CostRegNet(in_channels=self.backbone.out_channels, base_channels=self.backbone.out_channels)
        self.depth_head = Conv2d(1, self.backbone.out_channels, 3, stride=1, padding=1)
        self.fuse_conv = Conv2d(self.backbone.out_channels * 3, self.backbone.out_channels, 3, stride=1, padding=1)
        self.fuse_conv_c = nn.Sequential(
            Conv2d(self.backbone.out_channels, self.backbone.out_channels * 2, 5, stride=2, padding=2),
            Conv2d(self.backbone.out_channels * 2, self.backbone.out_channels * 4, 5, stride=2, padding=2),
            Conv2d(self.backbone.out_channels * 4, self.backbone.out_channels * 8, 5, stride=2, padding=2),
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

    def forward(self, data_dict, epoch=0):
        output_dict = {}
        # step 1. feature extraction
        A_imgs = data_dict['A_imgs'] #(B, N, C, H, W)
        B_imgs = data_dict['B_imgs'] #(B, N, C, H, W)
        batch, num, channel, height, width = A_imgs.shape
        A_imgs = A_imgs.view(batch * num, channel, height, width) #(B * N, C, H, W)
        B_imgs = B_imgs.view(batch * num, channel, height, width) #(B * N, C, H, W)
        imgs = torch.cat([A_imgs, B_imgs], dim=0) #(2 * B * N, C, H, W)

        features = self.backbone(imgs) #(2 * B * N, c, h, w)
        A_intrinsics = data_dict['A_intrinsics'] #(B, N, 4, 4)
        B_intrinsics = data_dict['B_intrinsics'] #(B, N, 4, 4)
        intrinsics = torch.cat([A_intrinsics, B_intrinsics], dim=0) #(2 * B, N, 4, 4)
        A_extrinsics = data_dict['A_extrinsics'] #(B, N, 4, 4)
        B_extrinsics = data_dict['B_extrinsics'] #(B, N, 4, 4)
        extrinsics = torch.cat([A_extrinsics, B_extrinsics], dim=0) #(2 * B, N, 4, 4)
        A_near_depth, A_far_depth = data_dict['A_depth_range'][0], data_dict['A_depth_range'][1]
        B_near_depth, B_far_depth = data_dict['B_depth_range'][0], data_dict['B_depth_range'][1]
        device, dtype = features.device, features.dtype
        A_depth_interval = (A_far_depth - A_near_depth) / (self.num_depth - 1)
        A_depth_range_samples = A_near_depth.unsqueeze(1) + (torch.arange(0, self.num_depth, device=device, dtype=dtype, requires_grad=False).reshape(1, -1) * A_depth_interval.unsqueeze(1)) #(B, D)
        A_depth_range_samples = A_depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, features.shape[2], features.shape[3]) #(B, D, H, W)
        B_depth_interval = (B_far_depth - B_near_depth) / (self.num_depth - 1)
        B_depth_range_samples = B_near_depth.unsqueeze(1) + (torch.arange(0, self.num_depth, device=device, dtype=dtype, requires_grad=False).reshape(1, -1) * B_depth_interval.unsqueeze(1)) #(B, D)
        B_depth_range_samples = B_depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, features.shape[2], features.shape[3]) #(B, D, H, W)
        depth_values = torch.cat([A_depth_range_samples, B_depth_range_samples], dim=0).to(torch.float32) #(2 * B, D, h, w)
        A_features = features[:features.shape[0] // 2].view(batch, num, -1, features.shape[2], features.shape[3]) #(B, N, c, h, w)
        B_features = features[features.shape[0] // 2:].view(batch, num, -1, features.shape[2], features.shape[3]) #(B, N, c, h, w)
        features = torch.cat([A_features, B_features], dim=0) #(2 * B, N, c, h, w)
        num_views = features.shape[1]

        ref_feature = features[:, 0] #(2 * B, c, h, w)
        src_features = [features[:, i + 1] for i in range(num_views - 1)] #[(2 * B, c, h, w), ...]
        ref_intr = intrinsics[:, 0] #(B, 4, 4)
        src_intrinsics = [intrinsics[:, i + 1] for i in range(num_views - 1)] #[(B, 4, 4), ...]
        ref_extr = extrinsics[:, 0] #(B, 4, 4)
        src_extrinsics = [extrinsics[:, i + 1] for i in range(num_views - 1)] #[(B, 4, 4), ...]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, self.num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_intr, src_extr in zip(src_features, src_intrinsics, src_extrinsics):
            #warpped features
            src_proj_new = src_extr.clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_intr[:, :3, :3], src_extr[:, :3, :4])
            ref_proj_new = ref_extr.clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_intr[:, :3, :3], ref_extr[:, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. cost volume regularization
        cost_reg, cost_feature = self.cost_regularization(volume_variance)
        prob_volume_pre = cost_reg.squeeze(1)
        prob_volume = F.softmax(prob_volume_pre * 5., dim=1)
        depth_pre, cost_feature, photometric_confidence = depth_regression_top2(prob_volume, cost_feature, depth_values=depth_values)
        # if self.training:
        near_depth = torch.cat([A_near_depth, B_near_depth]).to(torch.float32)[:,None,None]
        far_depth = torch.cat([A_far_depth, B_far_depth]).to(torch.float32)[:,None,None]
        if self.training:
            with torch.no_grad():
                # if stage_idx > 0:
                # range_min = near_depth
                # range_max = far_depth
                depth = torch.cat([data_dict['A_np_depth_ds'], data_dict['B_np_depth_ds']], dim=0).to(torch.float32)
                B,H,W = depth.shape
                mask = torch.where((near_depth <= depth) * (depth < far_depth), 1, 0).type(torch.long)#(B,H,W)
                new_interval = (depth_values[:,1,0,0] - depth_values[:,0,0,0]).view(depth_values.shape[0],1,1)#(B,1,1)
                x = depth.unsqueeze(1) - depth_values
                x = torch.where(x >= 0, x, torch.tensor(1e6, device=x.device))
                [gt_value_image, gt_index_image] = torch.min(x, dim=1)
                gt_index_image = torch.mul(mask, gt_index_image.type(torch.float))
                gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)
                gt_value_image = (1 - gt_value_image / new_interval).unsqueeze(1)
                gt_index_volume = torch.zeros_like(depth_values).scatter_(1, gt_index_image, gt_value_image)
                gt_index_image = (gt_index_image.type(torch.float) + 1).type(torch.long)
                gt_index_image = torch.mul(mask.unsqueeze(1), gt_index_image)
                gt_value_image = 1 - gt_value_image
                gt_index_volume = gt_index_volume.scatter_(1, gt_index_image, gt_value_image)
                error = ((depth - depth_pre) / (far_depth - near_depth)).abs()[mask.to(torch.bool)].mean()
                output_dict['abs_error'] = error
            if epoch < 5:
                depth = torch.cat([data_dict['A_np_depth_ds'], data_dict['B_np_depth_ds']], dim=0).to(torch.float32)
            elif epoch >= 5 and epoch < 10:
                if random.random() > 0.5:
                    depth = (depth_pre + torch.cat([data_dict['A_np_depth_ds'], data_dict['B_np_depth_ds']], dim=0).to(torch.float32)) / 2
                else:
                    fuse_mask = torch.randint(0,1,depth_pre.shape).to(depth_pre.device)
                    depth = depth_pre * fuse_mask + torch.cat([data_dict['A_np_depth_ds'], data_dict['B_np_depth_ds']], dim=0).to(torch.float32) * (1 - fuse_mask)
            else:
                if random.random() > 0.5:
                    depth = (4*depth_pre + torch.cat([data_dict['A_np_depth_ds'], data_dict['B_np_depth_ds']], dim=0).to(torch.float32)) / 5
                else:
                    fuse_mask = (torch.randint(0,4,depth_pre.shape) > 0).to(torch.float32).to(depth_pre.device)
                    depth = depth_pre * fuse_mask + torch.cat([data_dict['A_np_depth_ds'], data_dict['B_np_depth_ds']], dim=0).to(torch.float32) * (1 - fuse_mask)
        else:
            depth = depth_pre
        depth_normalize = (depth - near_depth) / (far_depth - near_depth)
        depth_feature = self.depth_head(depth_normalize[:,None,:,:].to(torch.float32))
        fuse_features_f = self.fuse_conv(torch.cat([features[:,0], cost_feature, depth_feature], dim=1)) #(2 * B, 3c, h, w)
        fuse_features_c = self.fuse_conv_c(fuse_features_f) #(2 * B, 8c, h/8, w/8)
        A_proj = data_dict['ref_extrinsics'].clone() #(B,4,4)
        B_proj = data_dict['src_extrinsics'].clone() #(B,4,4)
        A_proj[:, :3, :4] = torch.matmul(A_intrinsics[:, 0, :3, :3], A_proj[:, :3, :4])
        B_proj[:, :3, :4] = torch.matmul(B_intrinsics[:, 0, :3, :3], B_proj[:, :3, :4])
        proj = torch.cat([A_proj, B_proj], dim=0)
        position_f = _projection(cost_feature, torch.inverse(proj.cpu()).to(fuse_features_f.device), depth[:,None,...])
        position_c = position_f[:, :, 3::8, 3::8]

        if self.training:
            valid_pixel_num = torch.sum(mask, dim=[1,2]) + 1e-6
            cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-10), dim=1).squeeze(1) # B, 1, H, W
            masked_cross_entropy_image = torch.mul(mask, cross_entropy_image) # valid pixel
            masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
            masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)
            output_dict['masked_cross_entropy'] = masked_cross_entropy

            ref_points_c = position_c.permute(0,2,3,1)[0][mask[0, 3::8, 3::8] > 0.5].contiguous()
            src_points_c = position_c.permute(0,2,3,1)[1][mask[1, 3::8, 3::8] > 0.5].contiguous()
            ref_points_f = position_f.permute(0,2,3,1)[0][mask[0] > 0.5].contiguous()
            src_points_f = position_f.permute(0,2,3,1)[1][mask[1] > 0.5].contiguous()

            transform = data_dict['transform'][0].detach()
        else:
            ref_points_c = position_c.permute(0,2,3,1)[0][photometric_confidence[0, 3::8, 3::8] > min(0.1,photometric_confidence[0, 3::8, 3::8].mean())].contiguous()
            src_points_c = position_c.permute(0,2,3,1)[1][photometric_confidence[1, 3::8, 3::8] > min(0.1,photometric_confidence[1, 3::8, 3::8].mean())].contiguous()
            ref_points_f = position_f.permute(0,2,3,1)[0][photometric_confidence[0] > min(0.1,photometric_confidence[0].mean())].contiguous()
            src_points_f = position_f.permute(0,2,3,1)[1][photometric_confidence[1] > min(0.1,photometric_confidence[1].mean())].contiguous()
            ref_points_f_center = ref_points_f.mean(0,keepdim=True)
            src_points_f_center = src_points_f.mean(0,keepdim=True)
            ref_points_c = (ref_points_c - ref_points_f_center) * 3 / (A_far_depth - A_near_depth)
            src_points_c = (src_points_c - src_points_f_center) * 3 / (A_far_depth - A_near_depth)
            ref_points_f = (ref_points_f - ref_points_f_center) * 3 / (A_far_depth - A_near_depth)
            src_points_f = (src_points_f - src_points_f_center) * 3 / (A_far_depth - A_near_depth)

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points_f
        output_dict['src_points'] = src_points_f

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)
        if self.training:
            gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
            )

            output_dict['gt_node_corr_indices'] = gt_node_corr_indices
            output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        if self.training:
            if epoch < 5:
                ref_feats_c = fuse_features_c[0].reshape(fuse_features_c.shape[1], -1).transpose(1,0).contiguous()
                src_feats_c = fuse_features_c[1].reshape(fuse_features_c.shape[1], -1).transpose(1,0).contiguous()
            else:
                ref_feats_c = fuse_features_c.permute(0,2,3,1)[0][mask[0, 3::8, 3::8] > 0.5].contiguous()
                src_feats_c = fuse_features_c.permute(0,2,3,1)[1][mask[1, 3::8, 3::8] > 0.5].contiguous()
        else:
            ref_feats_c = fuse_features_c.permute(0,2,3,1)[0][photometric_confidence[0, 3::8, 3::8] > min(0.1,photometric_confidence[0, 3::8, 3::8].mean())].contiguous()
            src_feats_c = fuse_features_c.permute(0,2,3,1)[1][photometric_confidence[1, 3::8, 3::8] > min(0.1,photometric_confidence[1, 3::8, 3::8].mean())].contiguous()
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        if self.training:
            if epoch < 5:
                ref_feats_f = fuse_features_f[0].reshape(fuse_features_f.shape[1], -1).transpose(1,0).contiguous()
                src_feats_f = fuse_features_f[1].reshape(fuse_features_f.shape[1], -1).transpose(1,0).contiguous()
            else:
                ref_feats_f = fuse_features_f.permute(0,2,3,1)[0][mask[0] > 0.5].contiguous()
                src_feats_f = fuse_features_f.permute(0,2,3,1)[1][mask[1] > 0.5].contiguous()
        else:
            ref_feats_f = fuse_features_f.permute(0,2,3,1)[0][photometric_confidence[0] > min(0.1,photometric_confidence[0].mean())].contiguous()
            src_feats_f = fuse_features_f.permute(0,2,3,1)[1][photometric_confidence[1] > min(0.1,photometric_confidence[1].mean())].contiguous()
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / ref_feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            estimated_transform = registration_with_ransac_from_correspondences(
                (src_corr_points / 3 * (A_far_depth - A_near_depth) + src_points_f_center).to('cpu'),
                (ref_corr_points / 3 * (A_far_depth - A_near_depth) + ref_points_f_center).to('cpu'),
                distance_threshold=0.05 * (A_far_depth - A_near_depth) / 3,
                ransac_n=4,
                num_iterations=10000,
            )   

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = torch.from_numpy(estimated_transform).to(torch.float32).to(src_corr_points.device)

        return output_dict


def create_model(config):
    model = GeoTransformer(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
