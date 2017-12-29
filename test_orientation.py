import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
from tqdm import tqdm
from myutils.transform import abc2Rt
from sixd.params.dataset_params import get_dataset_params
from sixd.pysixd import renderer
from sixd.pysixd.inout import load_ply, load_info, load_gt
import model_orientation

img_name = sorted(os.listdir('data'))
images0 = []
images = []
for filename in tqdm(img_name):
    image = cv2.imread('data/' + filename)
    images0.append(cv2.resize(image, (200, int(200 * image.shape[0]/image.shape[1]))))
    images.append(cv2.resize(image, (64, 64)))

images = np.array(images).astype('float32')
images /= 255

y = []

model = model_orientation.get_model(images.shape[1:])

model.load_weights('ori_019_0.0345.h5')
y.append(model.predict(images).tolist())

y = np.mean(y, axis=0).tolist()
for i, yi in enumerate(y):
    a = yi[0] + 0.5
    b = (yi[1] + 0.5) * np.pi / 2
    c = yi[2] * np.pi / 2
    print(a, b, c)
    y[i] = abc2Rt(a, b, c, 400)

dataset_path = '/home/victorhuang/Desktop/pose/datasets/hinterstoisser/'
dp = get_dataset_params('hinterstoisser')
model = load_ply(dp['model_mpath'].format(2))
scene_info = load_info(dp['scene_info_mpath'].format(2))
K = scene_info[0]['cam_K']
for i, yi in enumerate(y):
    R = yi[0]
    t = yi[1]
    model_img = renderer.render(model, (640, 480), K, R, t, mode='rgb')
    model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Input', images0[i])
    cv2.imshow('Predicted', model_img)
    cv2.waitKey()
