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
    radius = max(xmax-xmin, ymax-ymin, zmax-zmin) / 2

    pivots = getPivots(xmin, xmax, ymin, ymax, zmin, zmax, pivot_step, 0, 0,
                       render_resize / render_crop, K, np.eye(3), np.array([0, 0, 100]), shrink=0.0)
    #pivots = getIcosahedronPivots(radius, 0, 0, 0, K, np.eye(3), np.array([0, 0, 100]))
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

        bb = getBoundingBox(model_img)
        u_center = (bb[0] + bb[2]) // 2
        v_center = (bb[1] + bb[3]) // 2
        pad = 3
        dim = max(bb[2] - bb[0], bb[3] - bb[1]) + pad
        model_img = model_img[v_center - dim // 2: v_center + dim // 2,
                              u_center - dim // 2: u_center + dim // 2]

        # pivots = getPivots(xmin, xmax, ymin, ymax, zmin, zmax, pivot_step,
        #                    img_w // 2 - render_crop // 2, img_h // 2 - render_crop // 2,
        #                    render_resize / render_crop, K, R, t, shrink=0.0)

        # pivots = getPivots(xmin, xmax, ymin, ymax, zmin, zmax, pivot_step,
        #                    u_center - dim // 2, v_center - dim // 2,
        #                    render_resize / dim, K, R, t, shrink=0.0)

        pivots = getIcosahedronPivots(radius, u_center - dim // 2, v_center - dim // 2,
                                      render_resize / dim, K, R, t)

        pivots = [p[1] for p in pivots]
        '''
        pivots_vis = np.zeros((96, 96, 3)).astype(np.float32)
        for pi, p in enumerate(pivots):
            u = int(p[0])
            v = int(p[1])
            if 0 <= u < 96 and 0 <= v < 96:
                pivots_vis[v][u] = np.array([1.0, 1.0, 1.0])
        cv2.imshow('pivots', pivots_vis)
        cv2.imshow('', model_img)
        cv2.waitKey()
        '''
        # model_img = model_img[img_h // 2 - render_crop // 2: img_h // 2 + render_crop // 2,
        #                       img_w // 2 - render_crop // 2: img_w // 2 + render_crop // 2]

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

    # for i in range(100):
    #        renderModelImage(0, views[i], lights[0])
    # Parallel(n_jobs=6, verbose=1)(delayed(renderModelImage)(i, view, light) \
    #                               for i, (view, light) in enumerate(zip(views, lights)))




