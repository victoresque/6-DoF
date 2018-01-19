import os
import json
import cv2
from PIL import Image
import numpy as np
from joblib import Parallel, delayed
from sixd.pysixd import renderer
from transform import *
from data import *
from params import *

for model_id in model_ids:
    model = getModel(model_id)
    K = getIntrinsic(model_id)

    model_pts = model['pts']
    xmin, xmax = np.min(model_pts[:, 0]), np.max(model_pts[:, 0])
    ymin, ymax = np.min(model_pts[:, 1]), np.max(model_pts[:, 1])
    zmin, zmax = np.min(model_pts[:, 2]), np.max(model_pts[:, 2])

    render_base = synth_base + 'orientation/{:02d}/render/'.format(model_id)
    ensureDir(render_base)
    patch_base = synth_base + 'orientation/{:02d}/patch/'.format(model_id)
    ensureDir(patch_base)
    def renderModelImage(im_id, view, light):
        R = view['R']
        t = view['t']

        model_img = renderer.render(model, (img_w, img_h), K, R, t, mode='rgb', shading='phong',
                                    ambient_weight=np.random.uniform(ambient_range[0], ambient_range[1]),
                                    light_src=light)

        sz = render_crop_size
        pivots = getPivots(xmin, xmax, ymin, ymax, zmin, zmax, pivot_step,
                           img_w // 2 - sz // 2, img_h // 2 - sz // 2,
                           render_resize / render_crop_size, K, R, t, shrink=0.2)

        model_img = cv2.GaussianBlur(model_img, (3, 3), 2)
        model_img = model_img[img_h // 2 - sz // 2: img_h // 2 + sz // 2,
                              img_w // 2 - sz // 2: img_w // 2 + sz // 2]

        model_img = cv2.resize(model_img, (render_resize, render_resize), interpolation=cv2.INTER_LINEAR)

        for pi, p in enumerate(pivots):
            i, j = int(p[1][1]), int(p[1][0])
            patch = model_img[i - patch_size // 2: i + patch_size // 2,
                              j - patch_size // 2: j + patch_size // 2]
            Image.fromarray(patch, 'RGBA').save(patch_base + '{:06d}_{:03d}.png'.format(im_id, pi))


        for p in pivots:
            model_img[int(p[1][1])][int(p[1][0])] = np.array([255, 0, 0, 255])

        # PIL supports saving RGBA to .png by default (tricky in OpenCV)
        Image.fromarray(model_img, 'RGBA').save(render_base + '{:06d}.png'.format(im_id))

        json.dump({
            'R': view['R'].tolist(),
            't': view['t'].tolist(),
            'pivots': pivots
        }, open(render_base + '{:06d}.json'.format(im_id), 'w'))

    views = getViews(view_count, view_radius, view_inplane_steps)
    lights = getRandomLights(len(views))
    print(len(views), 'views.')
    # renderModelImage(0, views[12], lights[12], render_base)
    Parallel(n_jobs=6, verbose=1)(delayed(renderModelImage)(i, view, light) \
                                  for i, (view, light) in enumerate(zip(views, lights)))




