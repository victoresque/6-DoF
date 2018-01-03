import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
from tqdm import tqdm
from myutils.transform import angle2Rot
from sixd.params.dataset_params import get_dataset_params
from sixd.pysixd import renderer
from sixd.pysixd.inout import load_ply, load_info, load_gt, load_yaml

images0 = []
images = []
model_id = 15

dp = get_dataset_params('hinterstoisser')
gt_path = dp['scene_gt_mpath'].format(model_id)
gt = load_yaml(gt_path)
img_count = len(gt) // 10
for i in tqdm(range(img_count)):
    bb = gt[i][0]['obj_bb']
    bb[0] = max(0, bb[0])
    bb[1] = max(0, bb[1])
    img_path = dp['test_rgb_mpath'].format(model_id, i)
    image = cv2.imread(img_path)
    image = image[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
    images0.append(cv2.resize(image, (200, int(200 * image.shape[0]/image.shape[1]))))
    image = cv2.resize(image, (96, 96))
    images.append(image)

images = np.array(images).astype('float32')
images /= 255

import model_orientation
model = model_orientation.get_model(images.shape[1:])
model.load_weights('ori_044_33.1586_32.9884.h5')

y = model.predict(images)
print(y)

y = y[0].tolist()
for i, yi in enumerate(y):
    rx = yi[0] * 2 * np.pi
    ry = yi[1] * 2 * np.pi
    rz = yi[2] * 2 * np.pi
    print(rx, ry, rz)
    y[i] = (angle2Rot(rx, ry, rz), np.array([0, 0, 400]))

model = load_ply(dp['model_mpath'].format(model_id))
scene_info = load_info(dp['scene_info_mpath'].format(model_id))
K = scene_info[0]['cam_K']
for i, yi in enumerate(y):
    R = yi[0]
    t = yi[1]
    model_img = renderer.render(model, (640, 480), K, R, t, mode='rgb')
    model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Input', images0[i])
    cv2.imshow('Predicted', model_img)
    cv2.waitKey()
