import os
import cv2
import numpy as np
from tqdm import tqdm
from sixd.params.dataset_params import get_dataset_params
from sixd.pysixd.inout import load_ply, load_info, load_gt


def getBackgrounds(bg_count):
    bg_base = '/home/victorhuang/Desktop/pose/datasets/VOC2012/JPEGImages/'
    bg = []
    bg_name = os.listdir(bg_base)
    np.random.shuffle(bg_name)
    bg_name = bg_name[:bg_count]
    for filename in tqdm(bg_name, 'Reading backgrounds'):
        bg.append(cv2.imread(bg_base + filename))
    return bg

def getModel(id):
    dp = get_dataset_params('hinterstoisser')
    model = load_ply(dp['model_mpath'].format(id))
    return model

def getIntrinsic(id, render_scale):
    dp = get_dataset_params('hinterstoisser')
    scene_info = load_info(dp['scene_info_mpath'].format(id))
    K = scene_info[0]['cam_K'] * render_scale
    K[2][2] = 1.
    return K

def ensureDir(path):
    if not os.path.exists(path):
        os.makedirs(path)