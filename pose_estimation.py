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

for model_id in model_ids:
    obj_model = getModel(model_id)
    dp = get_dataset_params('hinterstoisser')
    gt_path = dp['scene_gt_mpath'].format(model_id)
    gt = load_yaml(gt_path)
    K = getIntrinsic(model_id)
    img_count = len(gt) // 20

    render_base = synth_base + 'orientation/{:02d}/render/'.format(model_id)
    pivots = np.array([_[0] for _ in json.load(open(render_base + '000000.json', 'r'))['pivots']])

    images = []
    ratios = []
    images_origin = []  # in (u, v)
    for i in tqdm(range(img_count)):
        bb = gt[i][0]['obj_bb']

        img_path = dp['test_rgb_mpath'].format(model_id, i)
        image = cv2.imread(img_path)
        pad = 8
        dim = max(bb[2], bb[3])
        image = image[max(0, bb[1] - pad): bb[1] + bb[3] + pad, max(0, bb[0] - pad): bb[0] + bb[2] + pad]
        ratio = dim / render_resize
        image = cv2.resize(image, (int(bb[2] * ratio), int(bb[3] * ratio)))
        images.append(image)
        ratios.append(ratio)
        images_origin.append(np.array([max(0, bb[0] - pad), max(0, bb[1] - pad)]))
    '''
    for img in images:
        cv2.imshow('', img)
        cv2.waitKey()
    '''

    model = ConvolutionalAutoEncoder()
    model.load_state_dict(torch.load('models/model_epoch4.pth'))
    model.cuda()
    model.eval()

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


