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
    objectPts = np.array([_[0] for _ in json.load(open(render_base + '000000.json', 'r'))['pivots']])

    images = []
    ratios = []
    images_origin = []
    for i in tqdm(range(img_count)):
        bb = gt[i][0]['obj_bb']

        img_path = dp['test_rgb_mpath'].format(model_id, i)
        image = cv2.imread(img_path)
        pad = 8
        dim = max(bb[2], bb[3])
        bb[2] = dim
        bb[3] = dim
        image = image[max(0, bb[1] - pad): bb[1] + bb[3] + pad, max(0, bb[0] - pad): bb[0] + bb[2] + pad]
        image = cv2.resize(image, (render_resize, render_resize))
        images.append(image)
        ratios.append(dim / render_resize)
        images_origin.append(np.array([max(0, bb[1] - pad), max(0, bb[0] - pad)]))
    '''
    for img in images:
        cv2.imshow('', img)
        cv2.waitKey()
    '''

    model = ConvolutionalAutoEncoder()
    model.load_state_dict(torch.load('models/model_epoch2.pth'))
    model.cuda()
    model.eval()

    for i, img in enumerate(images):
        x = np.transpose(np.array([img]).astype(np.float), (0, 3, 1, 2)) / 255
        # x = np.array([img.flatten()]) / 255
        x = torch.FloatTensor(x).cuda()
        x = Variable(x)
        # output = model.forward(x).data.cpu().numpy().squeeze().reshape((-1, 2))
        output = model.forward(x).data.cpu().numpy()
        output = np.transpose(output, (0, 2, 3, 1))

        # imagePts = (output * render_resize + render_resize / 2) * ratios[i] + images_origin[i]

        # output = np.reshape(output, (render_resize, render_resize, 3))
        cv2.imshow('original', img)
        cv2.imshow('reconstructed', output[0])
        cv2.waitKey()

        '''
        ret = cv2.solvePnP(np.array(objectPts).astype(np.float32),
                           np.array(imagePts).astype(np.float32),
                           np.array(K), None)
        rvec = ret[1]
        R = cv2.Rodrigues(rvec)[0]
        t = ret[2]
        print(R)
        print(t)

        img = renderer.render(obj_model, (img_w, img_h), K, R, t, mode='rgb')
        cv2.imshow('Image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imshow('', images[i])
        cv2.waitKey()
        '''

