import os
import cv2
import numpy as np
from tqdm import tqdm
from keras.models import load_model
from myutils.render import Angle2Rot
from sixd.params.dataset_params import get_dataset_params
from sixd.pysixd import renderer
from sixd.pysixd.inout import load_ply, load_info, load_gt

model = load_model('ori_354_0.0112.h5')

img_name = sorted(os.listdir('data'))
images = []
for filename in tqdm(img_name):
    image = cv2.imread('data/' + filename)
    image = cv2.resize(image, (36, 36))
    images.append(image)

images = np.array(images)
y = model.predict(images)
for i, yi in enumerate(y):
    y[i] = Angle2Rot(i[0], i[1], i[2])

dataset_path = '/home/victorhuang/Desktop/pose/datasets/hinterstoisser/'
dp = get_dataset_params('hinterstoisser')
model = load_ply(dp['model_mpath'].format(1))
scene_info = load_info(dp['scene_info_mpath'].format(1))
K = scene_info[0]['cam_K']
for i, yi in enumerate(y):
    R = yi
    t = [[0], [0], [400]]
    model_img = renderer.render(model, (640, 480), K, R, t,
                                mode='rgb', ambient_weight=(np.random.rand() + 0.2))
    model_img = cv2.resize(model_img, (640, 480), interpolation=cv2.INTER_LINEAR)
    model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('W', model_img)
    cv2.waitKey()
