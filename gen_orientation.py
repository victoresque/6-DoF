import os
import json
import cv2
import multiprocessing
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sixd.pysixd import renderer
from myutils.transform import get_BB, getRandomViews
from myutils.data import getBackgrounds, get_model

view_count = 100000
bg_count = 10000
img_w = 640
img_h = 480

view_radius = 400
render_scale = 0.5
ambient_range = [0.25, 0.75]
light_shift = 200

bg = getBackgrounds(bg_count)
model_ids = [6, 8, 9, 13, 15]
for model_id in model_ids:
    model, K = get_model(model_id, render_scale)
    views = getRandomViews(view_count, view_radius)
    synth_base = '/home/victorhuang/Desktop/pose/algorithms/synthetic/'
    if not os.path.exists(synth_base + 'orientation/img/{:02d}'.format(model_id)):
        os.makedirs(synth_base + 'orientation/img/{:02d}'.format(model_id))
    if not os.path.exists(synth_base + 'orientation/nobg/{:02d}'.format(model_id)):
        os.makedirs(synth_base + 'orientation/nobg/{:02d}'.format(model_id))
    if not os.path.exists(synth_base + 'orientation/gt/{:02d}'.format(model_id)):
        os.makedirs(synth_base + 'orientation/gt/{:02d}'.format(model_id))

    # Parallel version of model rendering
    def rende_model_img(im_id, view):
        R = view['R']
        t = view['t']
        model_img = renderer.render(model, (int(img_w * render_scale),
                                            int(img_h * render_scale)), K, R, t,
                                    mode='rgb',
                                    ambient_weight=np.random.uniform(ambient_range[0], ambient_range[1]),
                                    light_src=[np.random.uniform(-light_shift, light_shift),
                                               np.random.uniform(-light_shift, light_shift), 0])
        model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
        li, ti, ri, bi = get_BB(model_img)
        li, ti, ri, bi = int(li * img_w), int(ti * img_h), int(ri * img_w), int(bi * img_h)
        model_img = model_img[ti-1:bi+2, li-1:ri+2]

        bg_img = cv2.resize(bg[im_id % bg_count], (img_w, img_h))

        model_img_w = model_img.shape[1]
        model_img_h = model_img.shape[0]
        mask = np.zeros((model_img_h + 2, model_img_w + 2), np.uint8)
        cv2.floodFill(model_img, mask, (0, 0), (0, 0, 0), upDiff=(1, 1, 1),
                      flags=cv2.FLOODFILL_FIXED_RANGE | 4 | (255 << 8))

        final_l = int(np.random.rand() * (img_w - model_img_w))
        final_r = final_l + model_img_w
        final_t = int(np.random.rand() * (img_h - model_img_h))
        final_b = final_t + model_img_h

        mask = mask[1:-1, 1:-1]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        bg_img = cv2.resize(cv2.bitwise_and(bg_img[final_t:final_b, final_l:final_r], mask) + model_img,
                            (96, 96), interpolation=cv2.INTER_LINEAR)

        model_img = cv2.resize(model_img, (96, 96))
        cv2.imwrite(synth_base + 'orientation/nobg/{:02d}/{:06d}.png'.format(model_id, im_id), model_img)
        cv2.imwrite(synth_base + 'orientation/img/{:02d}/{:06d}.png'.format(model_id, im_id), bg_img)

        view['R'] = view['R'].tolist()
        view['t'] = view['t'].tolist()
        with open(synth_base + 'orientation/gt/{:02d}/{:06d}.json'.format(model_id, im_id), 'w') as f:
            json.dump(view, f)

    Parallel(n_jobs=8, verbose=1)(delayed(rende_model_img)(i, view) for i, view in enumerate(views))
