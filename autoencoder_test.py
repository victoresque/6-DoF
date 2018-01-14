import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
from tqdm import tqdm
from sixd.params.dataset_params import get_dataset_params
from sixd.pysixd.inout import load_yaml

images0 = []
images = []
model_id = 6

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
    # images0.append(cv2.resize(image, (200, int(200 * image.shape[0]/image.shape[1]))))
    image = cv2.resize(image, (96, 96))
    images0.append(image)
    images.append(image)

images = np.array(images).astype('float32') / 255

from autoencoder_model import get_model
model = get_model(images.shape[1:])
model.load_weights('ori_109_0.0419_0.0383.h5')

images = model.predict(images).squeeze()
for i, image in enumerate(images):
    cv2.imshow('Input', images0[i])
    cv2.imshow('Output', image)
    cv2.waitKey()
