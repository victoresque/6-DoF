import cv2
import numpy as np
from PIL import Image
from params import *

def randomCrop(img, dim):
    img = cv2.resize(img, (dim * 8, dim * 8))
    L = int(np.random.rand() * (img.shape[1] - dim))
    T = int(np.random.rand() * (img.shape[0] - dim))
    return img[T:T + dim, L:L + dim]


def mergeImage(img, bg):
    img = Image.fromarray(img, 'RGBA')
    bg = Image.fromarray(bg, 'RGB').convert('RGBA')
    return Image.alpha_composite(bg, img).convert('RGB')

def randomPaste(img, backgrounds):
    bg = backgrounds[np.random.randint(0, len(backgrounds))]
    bg = randomCrop(bg, patch_size)
    return np.asarray(mergeImage(img, bg))

def samplePatch(img_id, img, cor, dim, stride):
    patches = []
    patches_info = []
    for i_center in range(dim // 2, img.shape[0] - dim // 2, stride):
        for j_center in range(dim // 2, img.shape[1] - dim // 2, stride):
            patches.append(img[i_center - dim // 2: i_center + dim // 2,
                               j_center - dim // 2: j_center + dim // 2])
            patches_info.append({'img_id': img_id, 'cor': cor[i_center][j_center].tolist()})
    return patches, patches_info

def sampleRGBDPatch(img_id, img, dep, f, cor, metric_dim, stride):
    patches = []
    patches_info = []
    for i_center in range(0, img.shape[0], stride):
        for j_center in range(0, img.shape[1], stride):
            z = dep[i_center][j_center]
            if z == 0:
                continue
            dim = int(metric_dim * f / z)
            if i_center - dim // 2 < 0 or i_center + dim // 2 > img.shape[0] \
                    or j_center - dim // 2 < 0 or j_center + dim // 2 > img.shape[1]:
                continue
            patch_img = img[i_center - dim // 2: i_center + dim // 2,
                           j_center - dim // 2: j_center + dim // 2]
            patch_dep = dep[i_center - dim // 2: i_center + dim // 2,
                           j_center - dim // 2: j_center + dim // 2] - z
            patch_dep = np.clip(patch_dep, -metric_dim, metric_dim)
            patch_img = patch_img / 255
            patch_dep = patch_dep / metric_dim
            patch_img = cv2.resize(patch_img, (patch_size, patch_size))
            patch_dep = cv2.resize(patch_dep, (patch_size, patch_size))
            patch = np.zeros((patch_img.shape[0], patch_img.shape[1], 4))
            patch[:, :, :3] = patch_img
            patch[:, :, 3] = patch_dep
            patches.append(patch)
            patches_info.append({'img_id': img_id, 'cor': cor[i_center][j_center].tolist()})
    return patches, patches_info
