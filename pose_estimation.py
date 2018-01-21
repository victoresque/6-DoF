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
    img_count = len(gt) // 20

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
        v_center = int(bb[1] + bb[3] / 2)
        u_center = int(bb[0] + bb[2] / 2)
        ratio = dim / render_resize
        image = image[v_center - dim // 2: v_center + dim // 2, u_center - dim // 2: u_center + dim // 2]
        image = cv2.resize(image, (render_resize, render_resize))
        images.append(image)
        ratios.append(ratio)
        images_origin.append(np.array([u_center - dim // 2, v_center - dim // 2]))

    model = CNN()
    model.load_state_dict(torch.load('models/model_epoch010_loss_0.001257_val_0.001168.pth'))
    model.cuda()
    model.eval()
    '''
    bgs = getBackgrounds(1000)
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
    for i, img in enumerate(images):
        objectPts = pivots
        cv2.imshow('!!', img)

        x = np.transpose(np.array([img]).astype(np.float), (0, 3, 1, 2)) / 255
        x = Variable(torch.FloatTensor(x)).cuda()

        output = model.forward(x).data.cpu().numpy() * render_resize + render_resize / 2
        imagePts = np.reshape(output, (-1, 2))

        pivots_vis = np.zeros((96, 96)).astype(np.float32)
        for p in imagePts:
            u = int(p[0])
            v = int(p[1])
            if 0 <= u < 96 and 0 <= v < 96:
                pivots_vis[u][v] = 1.
        cv2.imshow('pivots', pivots_vis)

        print(objectPts)
        print(imagePts)

        imagePts = imagePts + np.array(images_origin[i])

        pnp = cv2.solvePnPRansac(np.array(objectPts).astype(np.float32),
                                 np.array(imagePts).astype(np.float32),
                                 np.array(K), None)
        rvec = pnp[1]
        R = cv2.Rodrigues(rvec)[0]
        t = pnp[2]
        print(R)
        print(t)

        img = renderer.render(obj_model, (img_w, img_h), K, R, t, mode='rgb')
        cv2.imshow('??', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imshow('!?', images0[i])
        cv2.waitKey()


    '''
    def getSyntheticData(path, with_info):
        images = []
        images_info = []
        filenames = sorted(os.listdir(path))
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
    
    patch_base = synth_base + 'orientation/{:02d}/patch/'.format(model_id)
    pivot_images = getSyntheticData(patch_base, False)
    pivot_images = pivot_images[:2048]
    pivot_encode = []
    pivot_encode_id = []
    for i, img in enumerate(tqdm(pivot_images)):
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        x = np.transpose(np.array([img]).astype(np.float), (0, 3, 1, 2)) / 255
        x = Variable(torch.FloatTensor(x)).cuda()
        output = model.forward(x).data.cpu().numpy()
        output = np.transpose(output, (0, 2, 3, 1))
        pivot_encode.append(model.encode(x).data.cpu().numpy().flatten())
        pivot_encode_id.append(i)

    knn = KNeighborsClassifier().fit(pivot_encode, pivot_encode_id)

    for i, img in enumerate(images):
        stride = 4
        patches = []
        patches_center = []  # in (u, v)
        for i_center in range(patch_size // 2, img.shape[0] - patch_size // 2, stride):
            for j_center in range(patch_size // 2, img.shape[1] - patch_size // 2, stride):
                    patches.append(img[i_center - patch_size // 2: i_center + patch_size // 2,
                                       j_center - patch_size // 2: j_center + patch_size // 2])
                    patches_center.append([j_center * ratios[i] + images_origin[i][0],
                                           i_center * ratios[i] + images_origin[i][1]])

        objectPts = []
        imagePts = []
        for j, patch in enumerate(patches):
            x = np.transpose(np.array([patch]).astype(np.float), (0, 3, 1, 2)) / 255
            x = Variable(torch.FloatTensor(x)).cuda()

            output = model.forward(x).data.cpu().numpy()
            encode = model.encode(x).data.cpu().numpy().flatten()
            output = np.transpose(output, (0, 2, 3, 1))

            k = 1
            pivot_dis, pivot_id = knn.kneighbors([encode], k)
            # pivot_id = knn.kneighbors([encode], k)[1][0]
            for ki in range(k):
                objectPts.append(pivots[pivot_id[0][ki] % 27])
                imagePts.append(patches_center[j])

        print(len(objectPts))
        print(len(imagePts))

        pnp = cv2.solvePnPRansac(np.array(objectPts).astype(np.float32),
                                 np.array(imagePts).astype(np.float32),
                                 np.array(K), None)
        rvec = pnp[1]
        R = cv2.Rodrigues(rvec)[0]
        t = pnp[2]
        print(R)
        print(t)

        img = renderer.render(obj_model, (img_w, img_h), K, R, t, mode='rgb')
        cv2.imshow('', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imshow('{}'.format(i), images[i])
        cv2.waitKey()
    '''

