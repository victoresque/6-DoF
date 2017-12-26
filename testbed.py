import math
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

m = load_ply(dp['model_mpath'].format(6))
scene_info = load_info(dp['scene_info_mpath'].format(6))
scene_gt = load_gt(dp['scene_gt_mpath'].format(6))

views = sample_views(64, radius=256, elev_range=(0, 0.5 * math.pi))
K = scene_info[0]['cam_K']

for im_id in range(len(scene_gt)):
    R = views[0][im_id]['R']
    t = views[0][im_id]['t']

    rgb = renderer.render(m, (640, 480), K, R, t, mode='rgb')
    l, t, r, b = get_BB(rgb)

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = draw_BB(rgb/256, l, t, r, b)
    cv2.imshow('Render', rgb)
    cv2.waitKey()

