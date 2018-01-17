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

bg = getBackgrounds(bg_count)
for model_id in model_ids:
    obj_model = getModel(model_id)
    dp = get_dataset_params('hinterstoisser')
    gt_path = dp['scene_gt_mpath'].format(model_id)
    gt = load_yaml(gt_path)
    img_count = len(gt) // 10
    images = []
    patch_dims = []
    patch_origins = []
    for i in tqdm(range(img_count)):
        bb = gt[i][0]['obj_bb']

        img_path = dp['test_rgb_mpath'].format(model_id, i)
        image = cv2.imread(img_path)
        pad = 8
        image = image[max(0, bb[1] - pad): bb[1] + bb[3] + pad, max(0, bb[0] - pad): bb[0] + bb[2] + pad]
        images.append(image)
        patch_dims.append(int(max(bb[2], bb[3]) / 128 * patch_size * 1.5))
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
            if ext == '.png':
                patches.append(np.asarray(Image.open(path + filename)))
                patches[-1] = randomPaste(patches[-1], bg)
                patches[-1] = cv2.GaussianBlur(patches[-1], (3, 3), 1.0)
                patches[-1] = cv2.cvtColor(patches[-1], cv2.COLOR_BGR2RGB)
            if ext == '.json':
                patches_info.append(json.load(open(path + filename, 'r')))
        if with_info:
            return patches, patches_info
        else:
            return patches

    pivot_patches, pivot_patches_info = getSyntheticData(patch_base, True)

    model = SiameseNetwork()
    model.load_state_dict(torch.load('models/model_epoch46.pth'))
    model.cuda()
    model.eval()

    def samplePatches(img, dim, stride, u0, v0):
        patches = []
        screens = []
        for i_center in range(dim // 2, img.shape[0] - dim // 2, stride):
            for j_center in range(dim // 2, img.shape[1] - dim // 2, stride):
                patch = img[i_center - dim // 2: i_center + dim // 2, j_center - dim // 2: j_center + dim // 2]
                patch = cv2.resize(patch, (patch_size, patch_size))
                patches.append(patch)
                screens.append(np.array([u0 + j_center, v0 + i_center]))
        return patches, screens

    pivot_encode = []
    pivot_encode_id = []
    for i, pivot_patch in tqdm(enumerate(pivot_patches)):
        x = np.transpose(np.array([pivot_patch]).astype(np.float), (0, 3, 1, 2))
        x = torch.FloatTensor(x).cuda()
        x = Variable(x)
        enc = model.forward_once(x).data.cpu().numpy().squeeze()
        pivot_encode.append(enc)
        pivot_encode_id.append(i)
    
    np.save('pe.npy', pivot_encode)
    np.save('pei.npy', pivot_encode_id)

    pivot_encode = np.load('pe.npy')
    pivot_encode_id = np.load('pei.npy')

    knn = KNeighborsClassifier(n_neighbors=3).fit(pivot_encode, pivot_encode_id)
    K = getIntrinsic(model_id)

    for i, (img, dim) in enumerate(zip(images, patch_dims)):
        img_patches, img_patches_screens = samplePatches(img, dim, 4, patch_origins[i][1], patch_origins[i][0])
        objectPts = []
        imagePts = []
        for j, img_patch in enumerate(img_patches):
            img_patch = seq_brighten.augment_image(img_patch)
            x = np.transpose(np.array([img_patch]).astype(np.float), (0, 3, 1, 2))
            x = torch.FloatTensor(x).cuda()
            x = Variable(x)
            output = model.forward_once(x).data.cpu().numpy().squeeze()

            pivot_dis, pivot_id = knn.kneighbors([output], 1)
            if pivot_dis.squeeze() > 0.05:
                continue

            pivot_id = pivot_id.squeeze()
            cv2.imshow('patch', img_patch)
            cv2.imshow('pivot', pivot_patches[pivot_id])
            cv2.waitKey()
            objectPts.append(np.array(pivot_patches_info[pivot_id]['cor']))
            imagePts.append(img_patches_screens[j])

        ret = cv2.solvePnPRansac(np.array(objectPts).astype(np.float32),
                                 np.array(imagePts).astype(np.float32),
                                 np.array(K), None)
        rvec = ret[1]
        R = cv2.Rodrigues(rvec)[0]
        t = np.array([0, 0, 400])
        print(R)
        print(t)

        img = renderer.render(obj_model, (img_w, img_h), K, R, t, mode='rgb')
        cv2.imshow('Image', img)
        cv2.waitKey()
