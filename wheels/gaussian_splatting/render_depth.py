import os
import sys
from tqdm import tqdm

dataset_path = "ScanNet-GSReg/test"
# dataset_path = "data/outdoor"
scene_list = os.listdir(dataset_path)
for scene_name in tqdm(scene_list):
    scene_A_path = os.path.join(dataset_path, scene_name, "A")
    scene_B_path = os.path.join(dataset_path, scene_name, "B")
    A_command = 'CUDA_VISIBLE_DEVICES=1 python render.py --skip_test -m {model_A_path} --source_path {source_A_path}'.format(model_A_path=os.path.join(str(scene_A_path), 'output'), source_A_path=str(scene_A_path))
    B_command = 'CUDA_VISIBLE_DEVICES=1 python render.py --skip_test -m {model_B_path} --source_path {source_B_path}'.format(model_B_path=os.path.join(str(scene_B_path), 'output'), source_B_path=str(scene_B_path))
    os.system(A_command)
    os.system(B_command)
