import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
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
    K = getIntrinsic(model_id)

    pivots = np.load(synth_base + 'orientation/{:02d}/pivots.npy'.format(model_id))

    dp = get_dataset_params('hinterstoisser')
    gt_path = dp['scene_gt_mpath'].format(model_id)
    gt = load_yaml(gt_path)
    img_count = len(gt)

    images0 = []
    images = []
    ratios = []
    images_origin = []  # in (u, v)
    for i in tqdm(range(img_count)):
        bb = gt[i][0]['obj_bb']

        img_path = dp['test_rgb_mpath'].format(model_id, i)
        image = cv2.imread(img_path)
        images0.append(image)
        dim = int(max(bb[2], bb[3]))
        v_center = max(int(bb[1] + bb[3] / 2), dim // 2)
        u_center = max(int(bb[0] + bb[2] / 2), dim // 2)
        ratio = dim / render_resize
        image = image[v_center - dim // 2: v_center + dim // 2, u_center - dim // 2: u_center + dim // 2]
        image = cv2.resize(image, (render_resize, render_resize))
        images.append(image)
        ratios.append(ratio)
        images_origin.append(np.array([u_center - dim // 2, v_center - dim // 2]))

    model = CNN()
    model.load_state_dict(torch.load('models/model_epoch028_loss_0.001318_val_0.001352.pth'))
    model.cuda()
    model.eval()


    '''
    bgs = getBackgrounds(100)
    render_base = synth_base + 'orientation/{:02d}/render/'.format(model_id)
    def getSyntheticData(path, with_info):
        images = []
        images_info = []
        filenames = sorted(os.listdir(path))[:100]
        for filename in tqdm(filenames, 'Reading synthetic'):
            ext = os.path.splitext(filename)[1]
            if ext == '.png':
                images.append(np.asarray(Image.open(path + filename)))
            if ext == '.json':
                images_info.append(json.load(open(path + filename, 'r')))
        if with_info:
            return images, images_info
        else:
            return images
    images, images_info = getSyntheticData(render_base, True)
    images = [cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA) for image in images]
    images = np.array(images)
    images = [randomPaste(image, bgs) for image in images]
    '''


    visualize = False


    ae_sum = 0.
    re_sum = 0.
    te_sum = 0.
    for i, img in enumerate(images):
        objectPts = pivots
        x = np.transpose(np.array([img]).astype(np.float), (0, 3, 1, 2)) / 255
        x = Variable(torch.FloatTensor(x)).cuda()

        output = model.forward(x).data.cpu().numpy() * render_resize + render_resize / 2
        imagePts = np.reshape(output, (-1, 2))

        if visualize:
            pivots_vis = np.zeros((96, 96, 3)).astype(np.float32)
            for pi, p in enumerate(imagePts):
                u = int(p[0])
                v = int(p[1])
                if 0 <= u < 96 and 0 <= v < 96:
                    if 0 <= pi < pivot_step ** 2:
                        pivots_vis[v][u] = np.array([1.0, 1.0, 0.0])
                    if pivot_step ** 2 <= pi < 2 * pivot_step ** 2:
                        pivots_vis[v][u] = np.array([0.0, 1.0, 1.0])
                    if 2 * pivot_step ** 2 <= pi < 3 * pivot_step ** 2:
                        pivots_vis[v][u] = np.array([1.0, 0.0, 1.0])
            cv2.imshow('pivots', pivots_vis)

        imagePts = imagePts * ratios[i] + np.array(images_origin[i])

        pnp = cv2.solvePnPRansac(np.array(objectPts).astype(np.float32),
                           np.array(imagePts).astype(np.float32),
                           np.array(K), None)

        R = cv2.Rodrigues(pnp[1])[0]
        t = pnp[2].flatten()
        R_gt = np.array(gt[i][0]['cam_R_m2c']).reshape((3, 3))
        t_gt = np.array(gt[i][0]['cam_t_m2c'])

        ae = pose_error.add(R, t, R_gt, t_gt, obj_model)
        re = pose_error.re(R, R_gt)
        te = pose_error.te(t, t_gt)

        print('{:.2f}\t{:.2f}\t{:.2f}'.format(ae, re, te))

        if visualize or ae < 10:
            img = renderer.render(obj_model, (img_w, img_h), K, R, t, mode='rgb')
            cv2.imshow('??', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.imshow('!?', images0[i])
            cv2.waitKey()

        ae_sum += ae
        re_sum += re
        te_sum += te

    print('Average: {:.2f}\t{:.2f}\t{:.2f}'.format(ae_sum / img_count,
                                                   re_sum / img_count,
                                                   te_sum / img_count))