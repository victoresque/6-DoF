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
    for i_center in range(dim // 2, img.size[0] - dim // 2, stride):
        for j_center in range(dim // 2, img.size[1] - dim // 2, stride):
            if not np.array_equal(cor[i_center][j_center], cor[0][0]):
                patches.append(img.crop((j_center - dim // 2, i_center - dim // 2,
                                         j_center + dim // 2, i_center + dim // 2)))
                patches_info.append({'img_id': img_id, 'cor': cor[i_center][j_center].tolist()})
    return patches, patches_info