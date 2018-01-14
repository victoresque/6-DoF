import os
import json
import torch
import numpy as np
import imgaug
import cv2
from torch.autograd import Variable
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from params import *

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
    if len(pivot_rz) == 0 or gt['rz'] != pivot_rz[0]:
        pivot_rz.append(gt['rz'])
        pivot_rz_id.append(len(pivot_rz))
    else:
        break
print(len(pivot_rz))

vp_knn = KNeighborsClassifier(n_neighbors=3)
rz_knn = KNeighborsClassifier(n_neighbors=2)
vp_knn.fit(pivot_vp, pivot_vp_id)
rz_knn.fit(pivot_rz, pivot_rz_id)







