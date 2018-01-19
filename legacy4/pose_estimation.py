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
from sixd.pysixd.inout import load_ply, load_info, load_gt, load_yaml
from params import *
from myutils.data import getBackgrounds, getIntrinsic, getModel
from model import *
from sample import *
from augment import *

# bg = getBackgrounds(bg_count)
for model_id in model_ids:
    obj_model = getModel(model_id)
    dp = get_dataset_params('hinterstoisser')
    gt_path = dp['scene_gt_mpath'].format(model_id)
    gt = load_yaml(gt_path)
    img_count = len(gt)
    images = []
    depths = []
    patch_origins = []
    for i in tqdm(range(img_count)):
        bb = gt[i][0]['obj_bb']

        img_path = dp['test_rgb_mpath'].format(model_id, i)
        dep_path = dp['test_depth_mpath'].format(model_id, i)
        image = cv2.imread(img_path)
        depth = np.asarray(Image.open(dep_path))

        pad = 24
        dim = max(bb[2], bb[3])
        bb[2] = dim
        bb[3] = dim
        image = image[max(0, bb[1] - pad): bb[1] + bb[3] + pad, max(0, bb[0] - pad): bb[0] + bb[2] + pad]
        depth = depth[max(0, bb[1] - pad): bb[1] + bb[3] + pad, max(0, bb[0] - pad): bb[0] + bb[2] + pad]

        images.append(image)
        depths.append(depth)

        patch_origins.append([max(0, bb[1] - pad), max(0, bb[0] - pad)])
    '''
    for img in images:
        cv2.imshow('', img)
        cv2.waitKey()
    '''

    patch_base = synth_base + 'orientation/{:02d}/patch/'.format(model_id)

    def getSyntheticData(path, with_info):
        patches = []
        patches_info = []
        filenames = sorted(os.listdir(path))
        for filename in tqdm(filenames, 'Reading synthetic'):
            ext = os.path.splitext(filename)[1]
            if ext == '.npy':
                patches.append(np.load(path + filename))
            if ext == '.png':
                patches.append(np.asarray(Image.open(path + filename)))
                patches[-1] = cv2.cvtColor(patches[-1], cv2.COLOR_BGR2RGB)
            if ext == '.json':
                patches_info.append(json.load(open(path + filename, 'r')))
        if with_info:
            return patches, patches_info
        else:
            return patches

    pivot_patches, pivot_patches_info = getSyntheticData(patch_base, True)

    model = SiameseNetwork()
    model.load_state_dict(torch.load('models/model_epoch30.pth'))
    model.cuda()
    model.eval()

    pivot_encode = []
    pivot_encode_id = []
    for i, pivot_patch in tqdm(enumerate(pivot_patches)):
        x = np.transpose(np.array([pivot_patch]).astype(np.float), (0, 3, 1, 2))
        x = torch.FloatTensor(x).cuda()
        x = Variable(x)
        enc = model.forward_once(x).data.cpu().numpy().squeeze()
        pivot_encode.append(enc)
        pivot_encode_id.append(i)
    
    def sampleRGBDPatches(img_id, img, dep, f, metric_dim, stride, u0, v0):
        patches = []
        patches_screens = []
        for i_center in range(0, img.shape[0], stride):
            for j_center in range(0, img.shape[1], stride):
                z = dep[i_center][j_center]
                if z == 0:
                    continue
                dim = int(metric_dim * f / z)
                if i_center - dim // 2 < 0 or i_center + dim // 2 > img.shape[0] \
                        or j_center - dim // 2 < 0 or j_center + dim // 2 > img.shape[1]:
                    continue
                patch_img = img[i_center - dim // 2: i_center + dim // 2,
                            j_center - dim // 2: j_center + dim // 2]
                patch_dep = dep[i_center - dim // 2: i_center + dim // 2,
                            j_center - dim // 2: j_center + dim // 2] - z
                patch_dep = np.clip(patch_dep, -metric_dim, metric_dim)
                patch_img = patch_img / 255
                patch_dep = patch_dep / metric_dim
                patch_img = cv2.resize(patch_img, (patch_size, patch_size))
                patch_dep = cv2.resize(patch_dep, (patch_size, patch_size))
                patch = np.zeros((patch_img.shape[0], patch_img.shape[1], 4))
                patch[:, :, :3] = patch_img
                patch[:, :, 3] = patch_dep
                patches.append(patch)
                patches_screens.append(np.array([u0 + j_center, v0 + i_center]))
        return patches, patches_screens

    knn = KNeighborsClassifier(n_neighbors=3).fit(pivot_encode, pivot_encode_id)
    K = getIntrinsic(model_id)

    for i, (img, dep) in enumerate(zip(images, depths)):
        img_patches, img_patches_screens = \
            sampleRGBDPatches(i, img, dep, K[0][0], patch_metric_size, 8, patch_origins[i][1], patch_origins[i][0])

        objectPts = []
        encDis = []
        imagePts = []
        for j, img_patch in enumerate(img_patches):
            x = np.transpose(np.array([img_patch]).astype(np.float), (0, 3, 1, 2))
            x = torch.FloatTensor(x).cuda()
            x = Variable(x)
            output = model.forward_once(x).data.cpu().numpy().squeeze()

            pivot_dis, pivot_id = knn.kneighbors([output], 2)

            for k, id in enumerate(pivot_id[0]):
                # if k != 0: continue
                # cv2.imshow('patch', img_patch[:, :, :3])
                # cv2.imshow('pivot', pivot_patches[id][:, :, :3])
                objectPts.append(np.array(pivot_patches_info[id.squeeze()]['cor']))
                encDis.append(pivot_dis[0][k])
                imagePts.append(img_patches_screens[j])

        objectPts = np.array(objectPts).tolist()
        imagePts = np.array(imagePts).tolist()
        ss = sorted(zip(encDis, objectPts, imagePts))[:36]

        objectPts = [_[1] for _ in ss]
        imagePts = [_[2] for _ in ss]

        ret = cv2.solvePnPRansac(np.array(objectPts).astype(np.float32),
                                 np.array(imagePts).astype(np.float32),
                                 np.array(K), None, reprojectionError=2.0)
        R = cv2.Rodrigues(ret[1])[0]
        t = ret[2]
        print(R)
        print(t)

        img = renderer.render(obj_model, (img_w, img_h), K, R, t, mode='rgb')
        cv2.destroyAllWindows()
        cv2.imshow('{}'.format(i), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey()
