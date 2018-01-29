import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from sixd.params.dataset_params import get_dataset_params
from sixd.pysixd import renderer
from sixd.pysixd import pose_error
from sixd.pysixd.inout import load_ply, load_info, load_gt, load_yaml
from params import *
from data import getBackgrounds, getIntrinsic, getModel
from model import *
from sample import *
from augment import *
from transform import *

for model_id in model_ids:
    obj_model = getModel(model_id)

    model_pts = obj_model['pts']
    xmin, xmax = np.min(model_pts[:, 0]), np.max(model_pts[:, 0])
    ymin, ymax = np.min(model_pts[:, 1]), np.max(model_pts[:, 1])
    zmin, zmax = np.min(model_pts[:, 2]), np.max(model_pts[:, 2])
    radius = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2

    model_pts = obj_model['pts']
    K = getIntrinsic(model_id)

    dp = get_dataset_params('hinterstoisser')
    gt_path = dp['scene_gt_mpath'].format(model_id)
    gt = load_yaml(gt_path)

    img_count = len(gt)

    id_chosen = []
    images_chosen = []
    R_chosen = []
    anchors_chosen = []
    for i in tqdm(range(img_count)):
        img_path = dp['test_rgb_mpath'].format(model_id, i)
        image = cv2.imread(img_path)
        R_i = np.array(gt[i][0]['cam_R_m2c']).reshape((3, 3))
        t_i = np.array(gt[i][0]['cam_t_m2c']).reshape((3, 1))

        choose = True
        thres = 12
        for R in R_chosen:
            re = pose_error.re(R, R_i)
            if re < thres:
                choose = False
                break

        if choose:
            bb = gt[i][0]['obj_bb']
            dim = int(max(bb[2], bb[3]) * 1.2)
            v_center = max(int(bb[1] + bb[3] / 2), dim // 2)
            u_center = max(int(bb[0] + bb[2] / 2), dim // 2)
            image = image[v_center - dim // 2: v_center + dim // 2, u_center - dim // 2: u_center + dim // 2]
            image = cv2.resize(image, (render_resize, render_resize))

            u0 = u_center - dim // 2
            v0 = v_center - dim // 2

            anchors = getAnchors(xmin, xmax, ymin, ymax, zmin, zmax, anchor_step,
                               u0, v0, render_resize / dim, K, R_i, t_i, shrink=0.0)
            # anchors = getIcosahedronAnchors(radius, u0, v0, render_resize / dim, K, R_i, t_i)

            id_chosen.append(i)
            images_chosen.append(image)
            R_chosen.append(R_i)
            anchors_chosen.append(anchors)

    print(len(images_chosen), 'training images.')

    udpstyle_base = synth_base + 'udpstyle/{:02d}/'.format(model_id)
    for i in range(len(images_chosen)):
        cv2.imwrite(udpstyle_base + '{:06d}.png'.format(i), images_chosen[i])
        json.dump({
            'anchors': [p[1] for p in anchors_chosen[i]]
        }, open(udpstyle_base + '{:06d}.json'.format(i), 'w'))

