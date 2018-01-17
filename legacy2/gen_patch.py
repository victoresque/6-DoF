import os
import json
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sixd.pysixd import renderer
from myutils.transform import getBoundingBox, getViews
from myutils.data import getBackgrounds, getModel, getIntrinsic, ensureDir
from params import *
from sample import *

for model_id in model_ids:
    pivot_base = synth_base + 'orientation/{:02d}/pivot/'.format(model_id)

    def getSyntheticData(path):
        img = []
        cor = []
        filenames = sorted(os.listdir(path))
        for filename in tqdm(filenames, 'Reading synthetic:'):
            ext = os.path.splitext(filename)[1]
            if ext == '.png':
                img.append(Image.open(path + filename))
            if ext == '.npy':
                cor.append(np.load(path + filename))
        return img, cor

    model_img, model_cor = getSyntheticData(pivot_base)

    patches = []
    patches_info = []
    for i, img in tqdm(enumerate(model_img), 'Generating patches'):
        patches_, patches_info_ = samplePatch(i, img, model_cor[i], patch_size, patch_stride)
        # for patch in patches_:
        #     cv2.imshow('', cv2.cvtColor(np.asarray(patch), cv2.COLOR_BGR2RGB))
        #     cv2.waitKey()
        patches.extend(patches_)
        patches_info.extend(patches_info_)

    print(len(patches), 'patches.')

    patch_base = synth_base + 'orientation/{:02d}/patch/'.format(model_id)
    ensureDir(patch_base)
    for i, (patch, info) in tqdm(enumerate(zip(patches, patches_info))):
        # cv2.imwrite(patch_base + '{:06d}.png'.format(i), patch)
        patch.save(patch_base + '{:06d}.png'.format(i))
        json.dump(info, open(patch_base + '{:06d}.json'.format(i), 'w'))
