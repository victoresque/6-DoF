import os
import json
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sixd.params.dataset_params import get_dataset_params
from sixd.pysixd import renderer
from sixd.pysixd.inout import load_ply, load_info, load_gt, load_yaml
from myutils.transform import getBoundingBox, getViews
from myutils.data import getBackgrounds, getModel, getIntrinsic, ensureDir
from params import *
from sample import *

for model_id in model_ids:
    obj_model = getModel(model_id)
    K = getIntrinsic(model_id)
    K_inv = np.linalg.inv(K)

    dp = get_dataset_params('hinterstoisser')
    gt_path = dp['scene_gt_mpath'].format(model_id)
    gt = load_yaml(gt_path)
    img_count = len(gt)
    images = []
    cors = []
    patch_origins = []
    for i in tqdm(range(img_count)):
        if i % 20 != 0:
            continue
        bb = gt[i][0]['obj_bb']

        img_path = dp['test_rgb_mpath'].format(model_id, i)
        dep_path = dp['test_depth_mpath'].format(model_id, i)
        image = cv2.imread(img_path)
        depth = np.asarray(Image.open(dep_path))

        pad = 8
        dim = max(bb[2], bb[3])
        bb[2] = dim
        bb[3] = dim
        image = image[max(0, bb[1] - pad): bb[1] + bb[3] + pad, max(0, bb[0] - pad): bb[0] + bb[2] + pad]
        depth = depth[max(0, bb[1] - pad): bb[1] + bb[3] + pad, max(0, bb[0] - pad): bb[0] + bb[2] + pad]

        ratio = (dim + pad*2) / 96
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (96, 96), interpolation=cv2.INTER_NEAREST).tolist()

        images.append(image)

        R = np.reshape(gt[i][0]['cam_R_m2c'], (3, 3))
        t = np.reshape(gt[i][0]['cam_t_m2c'], (3, 1))
        obj2cam = np.zeros((4, 4))
        obj2cam[:3, :3] = R
        obj2cam[:3, 3:4] = t
        obj2cam[3, 3] = 1.
        cam2obj = np.linalg.inv(obj2cam)

        cor = depth
        for i, row in enumerate(depth):
            for j, dep in enumerate(row):
                cam_coord = np.matmul(K_inv, np.expand_dims([j * ratio + bb[0],
                                                             i * ratio + bb[1],
                                                             1], 1) * dep)
                obj_coord = np.matmul(cam2obj[:3, :3], cam_coord)
                obj_coord = obj_coord + cam2obj[:3, 3:4]
                cor[i][j] = obj_coord.squeeze()

        cors.append(np.array(cor))
        patch_origins.append([max(0, bb[1] - pad), max(0, bb[0] - pad)])

    model_img = np.array(images)
    model_cor = np.array(cors)

    patches = []
    patches_info = []
    for i, img in tqdm(enumerate(model_img), 'Generating patches'):
        patches_, patches_info_ = samplePatch(i, img, model_cor[i], patch_size, patch_stride)
        '''
        for patch in patches_:
            cv2.imshow('', np.asarray(patch))
            cv2.waitKey()
        '''
        patches.extend(patches_)
        patches_info.extend(patches_info_)

    print(len(patches), 'patches.')

    patch_base = synth_base + 'orientation/{:02d}/patch/'.format(model_id)
    ensureDir(patch_base)
    for i, (patch, info) in tqdm(enumerate(zip(patches, patches_info))):
        cv2.imwrite(patch_base + '{:06d}.png'.format(i), patch)
        json.dump(info, open(patch_base + '{:06d}.json'.format(i), 'w'))
