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

    def renderModelImage(im_id, view, light, path):
        R = view['R']
        t = view['t']

        model_img = renderer.render(model, (img_w, img_h), K, R, t, mode='rgb', shading='phong',
                                    ambient_weight=np.random.uniform(ambient_range[0], ambient_range[1]),
                                    light_src=light)

        sz = render_crop_size
        pivots = getPivots(xmin, xmax, ymin, ymax, zmin, zmax, pivot_step,
                           img_w // 2 - sz // 2, img_h // 2 - sz // 2,
                           render_resize / render_crop_size, K, R, t)

        model_img = cv2.GaussianBlur(model_img, (3, 3), 2)
        model_img = model_img[img_h // 2 - sz // 2: img_h // 2 + sz // 2,
                              img_w // 2 - sz // 2: img_w // 2 + sz // 2]

        model_img = cv2.resize(model_img, (render_resize, render_resize), interpolation=cv2.INTER_LINEAR)

        # PIL supports saving RGBA to .png by default (tricky in OpenCV)
        Image.fromarray(model_img, 'RGBA').save(path + '{:06d}.png'.format(im_id))

        json.dump({
            'R': view['R'].tolist(),
            't': view['t'].tolist(),
            'pivots': pivots
        }, open(path + '{:06d}.json'.format(im_id), 'w'))

    render_base = synth_base + 'orientation/{:02d}/render/'.format(model_id)
    ensureDir(render_base)

    views = getViews(view_count, view_radius, view_inplane_steps)
    lights = getRandomLights(len(views))
    print(len(views), 'views.')

    Parallel(n_jobs=6, verbose=1)(delayed(renderModelImage)(i, view, light, render_base) \
                                  for i, (view, light) in enumerate(zip(views, lights)))




