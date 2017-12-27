import math
import os
import json

import numpy as np
import cv2

from tqdm import tqdm
from sixd.params.dataset_params import get_dataset_params
from sixd.pysixd import renderer
from sixd.pysixd.inout import load_ply, load_info, load_gt
from sixd.pysixd.view_sampler import sample_views
from myutils.render import draw_BB, get_BB

dataset_path = '/home/victorhuang/Desktop/pose/datasets/hinterstoisser/'
dp = get_dataset_params('hinterstoisser')

voc_base = '/home/victorhuang/Desktop/pose/datasets/VOC2012/JPEGImages/'
bg = []
bg_name = os.listdir(voc_base)
np.random.shuffle(bg_name)

view_count = 1024
view_radius = 400
view_radius_ratio = 2
bg_count = 10000
img_count = 2048
img_width = 640
img_height = 480

bg_name = bg_name[:bg_count]
for filename in tqdm(bg_name):
    bg.append(cv2.imread(voc_base + filename))
for model_id in range(1, 2):
    if model_id == 3 or model_id == 7:
        # skip obj_03(bowl) and obj_07(mug)
        continue
    model = load_ply(dp['model_mpath'].format(model_id))
    scene_info = load_info(dp['scene_info_mpath'].format(model_id))
    scene_gt = load_gt(dp['scene_gt_mpath'].format(model_id))

    views = sample_views(view_count, radius=view_radius, elev_range=(-0.1 * math.pi, 0.5 * math.pi))[0]
    np.random.shuffle(views)
    K = scene_info[0]['cam_K']

    for im_id in tqdm(range(img_count)):
        bg_id = np.random.randint(bg_count)
        bg_img = np.copy(bg[bg_id])

        bg_img = cv2.resize(bg_img, (img_width, img_height))

        R = views[im_id % len(views)]['R']
        t = views[im_id % len(views)]['t']

        t = t * (np.random.rand() * view_radius_ratio + 1)
        t[0] += (np.random.rand() - 0.5) * t[2] * 0.85
        t[1] += (np.random.rand() - 0.5) * t[2] * 0.7

        model_img = renderer.render(model, (img_width, img_height), K, R, t,
                                    mode='rgb',
                                    ambient_weight=(np.random.rand()*1.2+0.2))
        model_img = cv2.resize(model_img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)

        # cv2.imshow('Model image', model_img)
        # cv2.waitKey()

        gt = {'R': R.tolist(), 't': t.tolist()}

        l, t, r, b = get_BB(model_img)
        li, ti, ri, bi = int(l*img_width), int(t*img_height), int(r*img_width), int(b*img_height)
        # model_img = cv2.GaussianBlur(model_img, (3, 3), 0)
        # model_img = cv2.bilateralFilter(model_img, 8, 32, 32)
        gt['bb'] = (l, t, r, b)

        model_img = cv2.cvtColor(model_img, cv2.COLOR_RGB2HSV)
        model_img[ti:bi + 1, li:ri + 1, 0] = (model_img[ti:bi + 1, li:ri + 1, 0]
                                              + int(np.random.rand() * 12 - 6)) % 255
        model_img[ti:bi + 1, li:ri + 1, 1] = (model_img[ti:bi + 1, li:ri + 1, 1]
                                              + int(np.random.rand() * 16 - 8)) % 255
        model_img = cv2.cvtColor(model_img, cv2.COLOR_HSV2RGB)

        h, w = model_img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        upThres = 16
        cv2.floodFill(model_img, mask, (0, 0), (0, 0, 0), upDiff=(upThres, upThres, upThres), flags=cv2.FLOODFILL_FIXED_RANGE | 4 | (255 << 8))
        cv2.floodFill(model_img, mask, (0, 0), (0, 0, 0), upDiff=(upThres, upThres, upThres), flags=cv2.FLOODFILL_FIXED_RANGE | 4 | (255 << 8))

        for i in range(ti, bi+1):
            for j in range(li, ri+1):
                pixel = model_img[i][j].astype(np.uint8)
                if not np.array_equal(pixel, [0, 0, 0]):
                    bg_img[i][j] = pixel

        # bg_img = draw_BB(bg_img, l, t, r, b)
        cv2.imwrite('synthetic/detection/img/'
                    'obj_{:02d}_{:05d}.png'.format(model_id, im_id), bg_img)

        with open('synthetic/detection/gt/'
                  'obj_{:02d}_{:05d}.json'.format(model_id, im_id), 'w') as f:
            json.dump(gt, f)

        # cv2.imshow('Final image', bg_img)
        # cv2.waitKey()

