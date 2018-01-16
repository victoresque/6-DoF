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

bg = getBackgrounds(bg_count)
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

    def randomCrop(img, dim):
        img = cv2.resize(img, (dim * 2, dim * 2))
        L = int(np.random.rand() * (img.shape[1] - dim))
        T = int(np.random.rand() * (img.shape[0] - dim))
        return img[T:T+dim, L:L+dim]

    def mergeImage(img, bg):
        bg = Image.fromarray(bg, 'RGB').convert('RGBA')
        return Image.alpha_composite(bg, img).convert('RGB')

    def samplePatch(img_id, img, cor, dim, stride):
        patches = []
        patches_info = []
        for i_center in range(dim // 2, img.size[0] - dim // 2, stride):
            for j_center in range(dim // 2, img.size[1] - dim // 2, stride):
                if not np.array_equal(cor[i_center][j_center], cor[0][0]):
                    patches.append(img.crop((j_center - dim // 2, i_center - dim // 2,
                                             j_center + dim // 2, i_center + dim // 2)))
                    patches_info.append({'img_id': img_id, 'cor': cor[i_center][j_center].tolist()})
        return patches, patches_info

    patches = []
    patches_info = []
    for i, img in tqdm(enumerate(model_img), 'Generating patches'):
        bg_img = bg[np.random.randint(0, len(bg))]
        img_ = mergeImage(img, randomCrop(bg_img, img.size[0]))
        cor_ = model_cor[i]
        patches_, patches_info_ = samplePatch(i, img_, cor_, patch_size, patch_stride)
        for i, patch in enumerate(patches_):
            patches_[i] = cv2.GaussianBlur(np.asarray(patch), (3, 3), 0.5)
        patches.extend(patches_)
        patches_info.extend(patches_info_)

    print(len(patches), 'patches.')

    patch_base = synth_base + 'orientation/{:02d}/patch/'.format(model_id)
    ensureDir(patch_base)
    for i, (patch, info) in tqdm(enumerate(zip(patches, patches_info))):
        cv2.imwrite(patch_base + '{:06d}.png'.format(i), patch)
        json.dump(info, open(patch_base + '{:06d}.json'.format(i), 'w'))


    rand_patch_base = synth_base + 'orientation/{:02d}/rand_patch/'.format(model_id)
    ensureDir(rand_patch_base)
    for i, img in tqdm(enumerate(range(len(patches))), 'Generating random patches'):
        patch = randomCrop(bg[np.random.randint(0, len(bg))], patch_size)
        patch = cv2.GaussianBlur(np.asarray(patch), (3, 3), 0.5)
        cv2.imwrite(rand_patch_base + '{:06d}.png'.format(i), patch)