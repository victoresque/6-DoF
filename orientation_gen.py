import os
import json
import cv2
import numpy as np
from joblib import Parallel, delayed
from sixd.pysixd import renderer
from myutils.transform import getBoundingBox, getViews
from myutils.data import getBackgrounds, getModel, getIntrinsic, ensureDir
from params import *

bg_count = 100
render_scale = 0.5
img_w = int(640 * render_scale)
img_h = int(480 * render_scale)
ambient_range = [0.5, 0.7]
light_shift = 200

view_radius = 400
pivot_count = 48
pivot_inplane_steps = 16
dense_count = 480
dense_inplane_steps = 32

# bg = getBackgrounds(bg_count)
model_ids = [6]  # [6, 8, 9, 13, 15]
for model_id in model_ids:
    model = getModel(model_id)
    K = getIntrinsic(model_id, render_scale)

    def renderModelImage(im_id, view, path):
        R = view['R']
        t = view['t']
        model_img = renderer.render(model, (img_w, img_h), K, R, t, mode='rgb',
                                    ambient_weight=np.random.uniform(ambient_range[0], ambient_range[1]),
                                    light_src=[np.random.uniform(-light_shift, light_shift),
                                               np.random.uniform(-light_shift, light_shift), 0])
        model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
        model_img = model_img[img_h//2-60: img_h//2+60, img_w//2-60: img_w//2+60]

        model_img_h = model_img.shape[0]
        model_img_w = model_img.shape[1]
        mask = np.zeros((model_img_h + 2, model_img_w + 2), np.uint8)
        cv2.floodFill(model_img, mask, (0, 0), (0, 0, 0), upDiff=(1, 1, 1),
                      flags=cv2.FLOODFILL_FIXED_RANGE | 4 | (255 << 8))

        '''
        bg_img = cv2.resize(bg[im_id % bg_count], (img_w, img_h))
        bg_l = int(np.random.rand() * (img_w - model_img_w))
        bg_r = bg_l + model_img_w
        bg_t = int(np.random.rand() * (img_h - model_img_h))
        bg_b = bg_t + model_img_h
        mask = mask[1:-1, 1:-1]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        bg_img = cv2.resize(cv2.bitwise_and(bg_img[bg_t:bg_b, bg_l:bg_r], mask) + model_img,
                            (96, 96), interpolation=cv2.INTER_LINEAR)
        '''

        model_img = cv2.resize(model_img, (96, 96))
        # cv2.imshow('', model_img)
        # cv2.waitKey()
        cv2.imwrite(path + '{:06d}.png'.format(im_id), model_img)

        view['R'] = view['R'].tolist()
        view['t'] = view['t'].tolist()
        view['vp'] = view['vp'].tolist()
        json.dump(view, open(path + '{:06d}.json'.format(im_id), 'w'))


    pivot_base = synth_base + 'orientation/pivot/{:02d}/'.format(model_id)
    ensureDir(pivot_base)
    views = getViews(pivot_count, view_radius, pivot_inplane_steps)
    Parallel(n_jobs=6, verbose=1)(delayed(renderModelImage)(i, view, pivot_base) for i, view in enumerate(views))

    dense_base = synth_base + 'orientation/dense/{:02d}/'.format(model_id)
    ensureDir(dense_base)
    views = getViews(dense_count, view_radius, dense_inplane_steps)
    Parallel(n_jobs=6, verbose=1)(delayed(renderModelImage)(i, view, dense_base) for i, view in enumerate(views))


