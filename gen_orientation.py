import math
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from sixd.params.dataset_params import get_dataset_params
from sixd.pysixd import renderer
from sixd.pysixd.inout import load_ply, load_info, load_gt
from sixd.pysixd.view_sampler import sample_views
from myutils.render import draw_BB, get_BB, Rot2Angle, getRandomView

dataset_path = '/home/victorhuang/Desktop/pose/datasets/hinterstoisser/'
voc_base = '/home/victorhuang/Desktop/pose/datasets/VOC2012/JPEGImages/'
dp = get_dataset_params('hinterstoisser')
bg = []
bg_name = os.listdir(voc_base)
np.random.shuffle(bg_name)

view_count = 50000
view_radius = 400
bg_count = 10000
img_count = view_count
img_width = 640
img_height = 480

bg_name = bg_name[:bg_count]
for filename in tqdm(bg_name):
    bg.append(cv2.imread(voc_base + filename))

for model_id in range(2, 3):
    if model_id == 3 or model_id == 7:
        # skip obj_03(bowl) and obj_07(mug)
        continue
    model = load_ply(dp['model_mpath'].format(model_id))
    scene_info = load_info(dp['scene_info_mpath'].format(model_id))
    scene_gt = load_gt(dp['scene_gt_mpath'].format(model_id))

    views = []
    # views = sample_views(view_count, radius=view_radius, elev_range=(0, 0.5 * math.pi))[0]
    for i in range(view_count):
        views.append(getRandomView(view_radius))
    print(len(views), 'views.')
    with open('/home/victorhuang/Desktop/pose/algorithms/synthetic/orientation/views.json', 'w') as f:
        _views = []
        for view in views:
            _views.append({'R': view['R'].tolist(),
                           't': view['t'].tolist(),
                           'u': view['u'],
                           'v': view['v']})
        json.dump(_views, f)

    K = scene_info[0]['cam_K']
    s = 0.5
    K = K * s
    K[2][2] = 1.

    model_imgs = []
    for view in tqdm(views):
        R, t = view['R'], view['t']
        model_img = renderer.render(model, (int(img_width * s), int(img_height * s)), K, R, t,
                                    mode='rgb',
                                    ambient_weight=np.random.uniform(0.5, 1),
                                    light_src=[np.random.uniform(-50, 50),
                                               np.random.uniform(-50, 50), 0])
        model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
        l, t, r, b = get_BB(model_img)
        li, ti, ri, bi = int(l * img_width), int(t * img_height), int(r * img_width), int(b * img_height)
        model_imgs.append(model_img[ti-1:bi+2, li-1:ri+2])

    for im_id in tqdm(range(img_count)):
        bg_id = np.random.randint(bg_count)
        bg_img = np.copy(bg[bg_id])
        bg_img = cv2.resize(bg_img, (img_width, img_height))

        model_img = model_imgs[im_id % len(model_imgs)]
        view = views[im_id % len(views)]
        gt = {'R': view['R'].tolist(),
              't': view['t'].tolist(),
              'u': view['u'], 'v': view['v']}
        # gt = {'view': im_id % len(views)}

        model_img = cv2.cvtColor(model_img, cv2.COLOR_RGB2HSV)
        model_img[:, :, 0] = ((model_img[:, :, 0] + int(np.random.uniform(-6, 6))) % 255).astype(np.uint8)
        model_img[:, :, 1] = ((model_img[:, :, 1] * np.random.uniform(0.9, 1))).astype(np.uint8)
        model_img[:, :, 2] = ((model_img[:, :, 2] * np.random.uniform(0.7, 1))).astype(np.uint8)
        model_img = cv2.cvtColor(model_img, cv2.COLOR_HSV2RGB)
        model_img_width = model_img.shape[1]
        model_img_height = model_img.shape[0]

        h, w = model_img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        upThres = 16
        cv2.floodFill(model_img, mask, (0, 0), (0, 0, 0), upDiff=(upThres, upThres, upThres), flags=cv2.FLOODFILL_FIXED_RANGE | 4 | (255 << 8))
        cv2.floodFill(model_img, mask, (0, 0), (0, 0, 0), upDiff=(upThres, upThres, upThres), flags=cv2.FLOODFILL_FIXED_RANGE | 4 | (255 << 8))

        final_l = int(np.random.rand() * (img_width - model_img_width))
        final_t = int(np.random.rand() * (img_height - model_img_height))

        mask = mask[1:-1, 1:-1]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        bg_img[final_t:final_t+model_img_height, final_l:final_l+model_img_width] \
            = cv2.bitwise_and(bg_img[final_t:final_t+model_img_height, final_l:final_l+model_img_width], mask)
        bg_img[final_t:final_t+model_img_height, final_l:final_l+model_img_width] \
            = bg_img[final_t:final_t+model_img_height, final_l:final_l+model_img_width] + model_img

        bb_w = model_img_width
        bb_h = model_img_height
        bb_shift = 0.25
        bb_dim = 0.2
        final_l = int(final_l + np.random.uniform(-bb_w // 2, bb_w // 2) * bb_shift)
        final_r = int(final_l + bb_w * (1 + np.random.rand() * bb_dim - bb_dim / 2))
        final_t = int(final_t + np.random.uniform(-bb_h // 2, bb_h // 2) * bb_shift)
        final_b = int(final_t + bb_h * (1 + np.random.rand() * bb_dim - bb_dim / 2))
        final_l = min(img_width, max(0, final_l))
        final_r = min(img_width, max(0, final_r))
        final_t = min(img_height, max(0, final_t))
        final_b = min(img_height, max(0, final_b))
        bg_img = bg_img[final_t:final_b+1, final_l:final_r+1]
        bg_img = cv2.resize(bg_img, (64, 64), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite('/home/victorhuang/Desktop/pose/algorithms/synthetic/orientation/img/'
                    'obj_{:02d}_{:05d}.png'.format(model_id, im_id), bg_img)

        with open('/home/victorhuang/Desktop/pose/algorithms/synthetic/orientation/gt/'
                  'obj_{:02d}_{:05d}.json'.format(model_id, im_id), 'w') as f:
            json.dump(gt, f)

        # cv2.imshow('Final image', bg_img)
        # cv2.waitKey()


