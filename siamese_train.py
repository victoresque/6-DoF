import os
import json
import torch
import numpy as np
import imgaug
import cv2
from torch.autograd import Variable
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sixd.pysixd import renderer
from params import *
from myutils.transform import lookAt

model_id = 6
pivot_base = synth_base + 'orientation/pivot/{:02d}/'.format(model_id)
dense_base = synth_base + 'orientation/dense/{:02d}/'.format(model_id)

def getSyntheticData(path):
    img, gt = [], []
    filenames = sorted(os.listdir(path))
    for filename in tqdm(filenames):
        ext = os.path.splitext(filename)[1]
        if ext == '.png':
            img.append(cv2.imread(path + filename))
        elif ext == '.json':
            gt.append(json.load(open(path + filename, 'r')))
    return img, gt


pivot_img, pivot_gt = getSyntheticData(pivot_base)
dense_img, dense_gt = getSyntheticData(dense_base)

pivot_vp, pivot_vp_id = [], []
pivot_rz, pivot_rz_id = [], []
for gt in pivot_gt:
    if len(pivot_vp) == 0 or gt['vp'] != pivot_vp[-1]:
        pivot_vp.append(gt['vp'])
        pivot_vp_id.append(len(pivot_vp))
for gt in pivot_gt:
    if len(pivot_rz) == 0 or gt['rz'] != pivot_rz[0][0]:
        pivot_rz.append([gt['rz']])
        pivot_rz_id.append(len(pivot_rz))
    else:
        break

knn_vp = KNeighborsClassifier(n_neighbors=3)
knn_rz = KNeighborsClassifier(n_neighbors=2)
knn_vp.fit(pivot_vp, pivot_vp_id)
knn_rz.fit(pivot_rz, pivot_rz_id)

n_pivot_vp = len(pivot_vp_id)
n_pivot_rz = len(pivot_rz_id)

def createPairs():
    x0_id = []
    x1_id = []
    label = []

    for di in tqdm(range(len(dense_img))):
        dense_vp_id = knn_vp.kneighbors([dense_gt[di]['vp']])[1].squeeze().tolist()
        dense_rz_id = knn_rz.kneighbors([[dense_gt[di]['rz']]])[1].squeeze().tolist()

        # True
        for pi in dense_vp_id:
            for ri in dense_rz_id:
                x0_id.append([pi, ri])
                x1_id.append(di)
                label.append(1)
        # False
        for pi in dense_vp_id:
            for ri in dense_rz_id:
                x0_id.append([pi, ri])
                x1_id.append(di)
                label.append(0)

    return x0_id, x1_id, label


a, b, c = createPairs()
print(len(a), len(b), len(c))