import os
import sys
from tqdm import tqdm

# dataset_path = "/workspace/data/dataset/test"
dataset_path = "/workspace/data/geotransformer/data/outdoor"
scene_list = os.listdir(dataset_path)
for scene_name in tqdm(scene_list):
    scene_A_path = os.path.join(dataset_path, scene_name, "part1")
    scene_B_path = os.path.join(dataset_path, scene_name, "part2")
    A_command = 'CUDA_VISIBLE_DEVICES=1 python render.py --skip_test -m '+ str(scene_A_path)
    B_command = 'CUDA_VISIBLE_DEVICES=1 python render.py --skip_test -m '+ str(scene_B_path)
    os.system(A_command)
    os.system(B_command)
