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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depths")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)
        """
        height = 480
        width = 640
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=rendering["render"].device),
                            torch.arange(0, width, dtype=torch.float32, device=rendering["render"].device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(width * height), x.view(width * height)
        xyz = torch.stack((x, y, torch.ones_like(x)))
        xyz = torch.unsqueeze(xyz, 0).repeat(1, 1, 1)
        from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
        K = torch.zeros(3,3)
        K[0,0] = fov2focal(view.FoVx, width)
        K[1,1] = fov2focal(view.FoVy, height)
        K[0,2] = width / 2
        K[1,2] = height / 2
        K[2,2] = 1.
        xyz_c = torch.matmul(torch.linalg.inv(K[None,:,:]).to(xyz.device), xyz)
        import cv2
        depth = rendering["render_depth"][0].view(1,1,-1)
        # depth  = torch.from_numpy(np.asarray(cv2.imread("./590.png", cv2.IMREAD_UNCHANGED)) / 1000).to(xyz.device).view(1,1,-1)
        xyz_c = xyz_c * depth - torch.from_numpy(view.T).to(xyz.device).view(1,-1,1)
        xyz_w = torch.matmul(torch.from_numpy(view.R).to(xyz.device).view(1,3,3), xyz_c)[0]
        from plyfile import PlyData,PlyElement
        points = xyz_w.cpu().numpy().transpose(1,0)
        colors = rendering["render"].view(3, -1).cpu().numpy().transpose(1,0) * 255
        alpha = rendering["render_alpha"].view(-1).cpu().numpy()
        index = alpha > 0.9
        points = points[index]
        colors = colors[index]
        # colors = np.asarray(cv2.pyrDown(cv2.imread("/workspace/data/dataset/train/scene0110_00/A/images/350.jpg"))).reshape(-1,3)
        points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=False).write("./scene_110_00_350_render.ply")
        # if idx == 0:
        #     depths = np.zeros((len(views), rendering.shape[0], rendering.shape[1]))
        # depths[idx] = rendering.cpu().numpy()
        """
        if idx == 0:
            depths = np.zeros((len(views), rendering["render_depth"][0].shape[0], rendering["render_depth"][0].shape[1]))
        depths[idx] = rendering["render_depth"][0].cpu().numpy()
        # np.save(os.path.join(render_path, view.image_name + ".npy"), rendering["render_depth"][0].cpu().numpy())
        # near = np.percentile(rendering["render_depth"][0].cpu().numpy(), 5)
        # far = np.percentile(rendering["render_depth"][0].cpu().numpy(), 95)
        # with open(os.path.join(render_path, view.image_name + ".txt"), 'w') as file:
        #     file.write(str(near) + ' ' + str(far))
        # gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if idx == len(views) - 1:
            np.save(os.path.join(render_path, "depths.npy"), depths)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)