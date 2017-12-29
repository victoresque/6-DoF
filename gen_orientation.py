import os
import json

import cv2
import multiprocessing
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from sixd.params.dataset_params import get_dataset_params
from sixd.pysixd import renderer
from sixd.pysixd.inout import load_ply, load_info, load_gt
from sixd.pysixd.view_sampler import sample_views
from myutils.transform import draw_BB, get_BB, Rot2Angle, getRandomView

bg_base = '/home/victorhuang/Desktop/pose/datasets/VOC2012/JPEGImages/'
bg = []
bg_name = os.listdir(bg_base)
np.random.shuffle(bg_name)

view_count = 100000
view_radius = 400
bg_count = 10000
img_w = 640
img_h = 480
render_scale = 0.5
ambient_range = [0.8, 1.2]
light_shift = 200
hue_shift = 8
bb_shift = 0.25
bb_dim = 0.2

bg_name = bg_name[:bg_count]
for filename in tqdm(bg_name, 'Reading backgrounds: '):
    bg.append(cv2.imread(bg_base + filename))

dp = get_dataset_params('hinterstoisser')
for model_id in range(2, 3):
    if model_id == 3 or model_id == 7:
        continue
    model = load_ply(dp['model_mpath'].format(model_id))

    scene_info = load_info(dp['scene_info_mpath'].format(model_id))
    scene_gt = load_gt(dp['scene_gt_mpath'].format(model_id))

    views = []
    for i in tqdm(range(view_count), 'Generating random views: '):
        views.append(getRandomView(view_radius))

    K = scene_info[0]['cam_K'] * render_scale
    K[2][2] = 1.

    # Parallel version of model rendering
    def rende_model_img(view):
        R, t = view['R'], view['t']
        model_img = renderer.render(model, (int(img_w * render_scale),
                                            int(img_h * render_scale)), K, R, t,
                                    mode='rgb',
                                    ambient_weight=np.random.uniform(ambient_range[0], ambient_range[1]),
                                    light_src=[np.random.uniform(-light_shift, light_shift),
                                               np.random.uniform(-light_shift, light_shift), 0])
        model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
        l, t, r, b = get_BB(model_img)
        li, ti, ri, bi = int(l * img_w), int(t * img_h), int(r * img_w), int(b * img_h)
        return view, model_img[ti - 1:bi + 2, li - 1:ri + 2]
    model_renders = Parallel(n_jobs=8, verbose=1)(delayed(rende_model_img)(view) for view in views)

    # Non-parallel version of model rendering
    '''
    model_imgs = []
    for view in tqdm(views):
        R, t = view['R'], view['t']
        model_img = renderer.render(model, (int(img_width * s), int(img_height * s)), K, R, t,
                                    mode='rgb',
                                    ambient_weight=np.random.uniform(0.5, 1),
                                    light_src=[np.random.uniform(-200, 200),
                                               np.random.uniform(-200, 200), 0])
        model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
        l, t, r, b = get_BB(model_img)
        li, ti, ri, bi = int(l * img_width), int(t * img_height), int(r * img_width), int(b * img_height)
        model_imgs.append(model_img[ti-1:bi+2, li-1:ri+2])
    '''

    for im_id in tqdm(range(view_count), 'Saving training data:'):
        bg_id = np.random.randint(bg_count)
        bg_img = cv2.resize(bg[bg_id], (img_w, img_h))

        view, model_img = model_renders[im_id]

        model_img = cv2.cvtColor(model_img, cv2.COLOR_RGB2HSV)
        model_img[:, :, 0] = ((model_img[:, :, 0] + int(np.random.uniform(-hue_shift, hue_shift))) % 255).astype(np.uint8)
        brighten = np.random.rand() > 0.4
        bri_ratio = np.random.rand() * 0.3
        dar_ratio = np.random.rand() * 0.2
        if brighten:
            model_img[:, :, 2] = ((model_img[:, :, 2] * (1 - bri_ratio) + 255 * bri_ratio)).astype(np.uint8)
        else:
            model_img[:, :, 2] = ((model_img[:, :, 2] * (1 - dar_ratio))).astype(np.uint8)
        model_img = cv2.cvtColor(model_img, cv2.COLOR_HSV2RGB)

        model_img_w = model_img.shape[1]
        model_img_h = model_img.shape[0]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(model_img, mask, (0, 0), (0, 0, 0), upDiff=(16, 16, 16), flags=cv2.FLOODFILL_FIXED_RANGE | 4 | (255 << 8))

        final_l = int(np.random.rand() * (img_w - model_img_w))
        final_t = int(np.random.rand() * (img_h - model_img_h))

        mask = mask[1:-1, 1:-1]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        bg_img[final_t:final_t + model_img_h, final_l:final_l + model_img_w] \
            = cv2.bitwise_and(bg_img[final_t:final_t + model_img_h, final_l:final_l + model_img_w], mask)
        bg_img[final_t:final_t + model_img_h, final_l:final_l + model_img_w] \
            = bg_img[final_t:final_t + model_img_h, final_l:final_l + model_img_w] + model_img

        final_l = int(final_l + np.random.uniform(-model_img_w // 2, model_img_w // 2) * bb_shift)
        final_r = int(final_l + model_img_w * (1 + np.random.rand() * bb_dim - bb_dim / 2))
        final_t = int(final_t + np.random.uniform(-model_img_h // 2, model_img_h // 2) * bb_shift)
        final_b = int(final_t + model_img_h * (1 + np.random.rand() * bb_dim - bb_dim / 2))

        bg_img = cv2.resize(bg_img[np.clip(final_t, 0, img_h):np.clip(final_b, 0, img_h) + 1, \
                                   np.clip(final_l, 0, img_w):np.clip(final_r, 0, img_w) + 1],
                            (96, 96), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite('/home/victorhuang/Desktop/pose/algorithms/synthetic/orientation/img/'
                    'obj_{:02d}_{:05d}.png'.format(model_id, im_id), bg_img)

        view['R'] = view['R'].tolist()
        view['t'] = view['t'].tolist()
        with open('/home/victorhuang/Desktop/pose/algorithms/synthetic/orientation/gt/'
                  'obj_{:02d}_{:05d}.json'.format(model_id, im_id), 'w') as f:
            json.dump(view, f)
