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
    depths = []
    cors = []
    patch_origins = []
    for i in tqdm(range(img_count)):
        if i % 50 != 0:
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

        images.append(image)
        depths.append(depth)

        R = np.reshape(gt[i][0]['cam_R_m2c'], (3, 3))
        t = np.reshape(gt[i][0]['cam_t_m2c'], (3, 1))
        obj2cam = np.zeros((4, 4))
        obj2cam[:3, :3] = R
        obj2cam[:3, 3:4] = t
        obj2cam[3, 3] = 1.
        cam2obj = np.linalg.inv(obj2cam)

        cor = np.zeros(image.shape).tolist()
        for i, row in enumerate(depth):
            for j, dep in enumerate(row):
                cam_coord = np.matmul(K_inv, np.expand_dims([j + bb[0],
                                                             i + bb[1],
                                                             1], 1) * dep)
                obj_coord = np.matmul(cam2obj[:3, :3], cam_coord)
                obj_coord = obj_coord + cam2obj[:3, 3:4]
                cor[i][j] = obj_coord.squeeze()

        cors.append(np.array(cor))
        patch_origins.append([max(0, bb[1] - pad), max(0, bb[0] - pad)])

    model_img = np.array(images)
    model_dep = np.array(depths)
    model_cor = np.array(cors).squeeze()

    patches = []
    patches_info = []
    for i, (img, dep) in enumerate(tqdm(zip(model_img, model_dep), 'Generating patches')):
        # patches_, patches_info_ = samplePatch(i, img, model_cor[i], patch_size, patch_stride)
        patches_, patches_info_ = sampleRGBDPatch(i, img, dep, K[0][0], model_cor[i], patch_metric_size, patch_stride)
        '''
        for patch in patches_:
            print(patch.shape)
            cv2.imshow('', patch)
            cv2.waitKey()
        '''
        patches.extend(patches_)
        patches_info.extend(patches_info_)

    print(len(patches), 'patches.')

    patch_base = synth_base + 'orientation/{:02d}/patch/'.format(model_id)
    ensureDir(patch_base)
    for i, (patch, info) in tqdm(enumerate(zip(patches, patches_info))):
        np.save(patch_base + '{:06d}.npy'.format(i), patch)
        json.dump(info, open(patch_base + '{:06d}.json'.format(i), 'w'))
