import os
import cv2
import numpy as np
from tqdm import tqdm
from sixd.params.dataset_params import get_dataset_params
from sixd.pysixd.inout import load_ply, load_info, load_gt


def getBackgrounds(bg_count):
    # bg_base = '/home/victorhuang/Desktop/pose/datasets/VOC2012/JPEGImages/'
    bg_base = '/home/victorhuang/Desktop/pose/datasets/hinterstoisser/test/01/rgb/'
    bg = []
    bg_name = os.listdir(bg_base)
    np.random.shuffle(bg_name)
    bg_name = bg_name[:bg_count]
    for filename in tqdm(bg_name, 'Reading backgrounds'):
        bg.append(cv2.imread(bg_base + filename)[200:400, 200:400])
        # bg.append(cv2.cvtColor(cv2.imread(bg_base + filename), cv2.COLOR_BGR2RGB))
    return bg

def getModel(id):
    dp = get_dataset_params('hinterstoisser')
    model = load_ply(dp['model_mpath'].format(id))
    return model

def getIntrinsic(id):
    dp = get_dataset_params('hinterstoisser')
    scene_info = load_info(dp['scene_info_mpath'].format(id))
    K = scene_info[0]['cam_K']
    return K

def ensureDir(path):
    if not os.path.exists(path):
        os.makedirs(path)