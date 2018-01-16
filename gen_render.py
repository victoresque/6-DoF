import os
import json
import cv2
from PIL import Image
import numpy as np
from joblib import Parallel, delayed
from sixd.pysixd import renderer
from myutils.transform import getBoundingBox, getViews, getLights
from myutils.data import getBackgrounds, getModel, getIntrinsic, ensureDir
from params import *

for model_id in model_ids:
    model = getModel(model_id)
    K = getIntrinsic(model_id)
    K_inv = np.linalg.inv(K)

    def renderModelImage(im_id, view, light, path):
        R, t = view['R'], view['t']
        obj2cam = np.zeros((4, 4))
        obj2cam[:3, :3] = R
        obj2cam[:3, 3:4] = t
        obj2cam[3, 3] = 1.
        cam2obj = np.linalg.inv(obj2cam)

        model_img, model_dep = renderer.render(model, (img_w, img_h), K, R, t, mode='rgb+depth',
                                               ambient_weight=np.random.uniform(ambient_range[0], ambient_range[1]),
                                               light_src=light)
        sz = 300
        model_img = model_img[img_h // 2 - sz // 2: img_h // 2 + sz // 2, img_w // 2 - sz // 2: img_w // 2 + sz // 2]
        model_dep = model_dep[img_h // 2 - sz // 2: img_h // 2 + sz // 2, img_w // 2 - sz // 2: img_w // 2 + sz // 2]

        rsz = 128
        model_img = cv2.resize(model_img, (rsz, rsz), interpolation=cv2.INTER_NEAREST)
        model_dep = cv2.resize(model_dep, (rsz, rsz), interpolation=cv2.INTER_NEAREST).tolist()

        for i, row in enumerate(model_dep):
            for j, dep in enumerate(row):
                cam_coord = np.matmul(K_inv, np.expand_dims([int(j * sz / rsz) + img_w // 2 - sz // 2,
                                                             int(i * sz / rsz) + img_h // 2 - sz // 2,
                                                             1], 1) * dep)
                obj_coord = np.matmul(cam2obj[:3, :3], cam_coord)
                obj_coord = obj_coord + cam2obj[:3, 3:4]
                model_dep[i][j] = obj_coord.squeeze()

        model_dep = np.array(model_dep)
        model_img = Image.fromarray(model_img, 'RGBA')

        model_img.save(path + '{:06d}.png'.format(im_id))
        np.save(path + '{:06d}.npy'.format(im_id), model_dep)

        view['R'] = view['R'].tolist()
        view['t'] = view['t'].tolist()
        view['vp'] = view['vp'].tolist()
        json.dump(view, open(path + '{:06d}.json'.format(im_id), 'w'))


    pivot_base = synth_base + 'orientation/{:02d}/pivot/'.format(model_id)
    ensureDir(pivot_base)
    views = getViews(pivot_count, view_radius, pivot_inplane_steps)
    lights = getLights(pivot_count * pivot_inplane_steps)
    Parallel(n_jobs=6, verbose=1)(delayed(renderModelImage)(i, view, light, pivot_base) \
                                  for i, (view, light) in enumerate(zip(views, lights)))

    '''
    dense_base = synth_base + 'orientation/{:02d}/dense/'.format(model_id)
    ensureDir(dense_base)
    views = getViews(dense_count, view_radius, dense_inplane_steps)
    lights = getLights(dense_count * dense_inplane_steps)
    Parallel(n_jobs=6, verbose=1)(delayed(renderModelImage)(i, view, light, dense_base)
                                  for i, (view, light) in enumerate(zip(views, lights)))
    '''



