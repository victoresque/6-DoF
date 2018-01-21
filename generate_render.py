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

    pivots = getPivots(xmin, xmax, ymin, ymax, zmin, zmax, pivot_step,
                       img_w // 2 - render_crop // 2, img_h // 2 - render_crop // 2,
                       render_resize / render_crop, K, np.eye(3), np.array([0, 0, 100]), shrink=0.0)
    pivots = np.array([p[0] for p in pivots])
    np.save(synth_base + 'orientation/{:02d}/pivots.npy'.format(model_id), pivots)

    render_base = synth_base + 'orientation/{:02d}/render/'.format(model_id)
    ensureDir(render_base)
    def renderModelImage(im_id, view, light):
        np.random.seed(im_id)

        R = view['R']
        t = view['t']
        t = t + np.array([[np.random.uniform(t_shift[0][0], t_shift[0][1])],
                          [np.random.uniform(t_shift[1][0], t_shift[1][1])],
                          [np.random.uniform(t_shift[2][0], t_shift[2][1])]])

        model_img = renderer.render(model, (img_w, img_h), K, R, t, mode='rgb', shading='phong',
                                    ambient_weight=np.random.uniform(ambient_range[0], ambient_range[1]),
                                    light_src=light)

        pivots = getPivots(xmin, xmax, ymin, ymax, zmin, zmax, pivot_step,
                           img_w // 2 - render_crop // 2, img_h // 2 - render_crop // 2,
                           render_resize / render_crop, K, R, t, shrink=0.0)

        pivots = [p[1] for p in pivots]

        model_img = model_img[img_h // 2 - render_crop // 2: img_h // 2 + render_crop // 2,
                              img_w // 2 - render_crop // 2: img_w // 2 + render_crop // 2]

        model_img = cv2.resize(model_img, (render_resize, render_resize), interpolation=cv2.INTER_LINEAR)

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

    # renderModelImage(0, views[0], lights[0])
    Parallel(n_jobs=6, verbose=1)(delayed(renderModelImage)(i, view, light) \
                                  for i, (view, light) in enumerate(zip(views, lights)))




