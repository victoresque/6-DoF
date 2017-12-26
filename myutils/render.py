import numpy as np
import cv2
from sklearn.preprocessing import normalize

def draw_BB(image, l, t, r, b):
    out = np.copy(image)
    lt = (int(l * image.shape[1]), int(t * image.shape[0]))
    rb = (int(r * image.shape[1]), int(b * image.shape[0]))
    cv2.rectangle(out, lt, rb, (0, 255, 0), 2)
    return out

def get_BB(image):
    a = np.where(image != 0)
    return np.min(a[1])/640, np.min(a[0])/480, \
           np.max(a[1])/640, np.max(a[0])/480

def RT2to4x4(R, t):
    pose = np.zeros((4, 4))
    np.copyto(pose[0:3, 0:3], R)
    np.copyto(pose[0:3, 3:4], t)
    return pose


def lookAt(src, dst):
    src = np.array(src)
    dst = np.array(dst)
    forward = normalize([dst - src]).flatten()
    right = normalize([np.cross(forward, [0, 0, 1])]).flatten()
    down = normalize([np.cross(forward, right)]).flatten()

    camToWorld = np.zeros((4, 4))

    camToWorld[0][0] = right[0]
    camToWorld[1][0] = right[1]
    camToWorld[2][0] = right[2]
    camToWorld[0][1] = down[0]
    camToWorld[1][1] = down[1]
    camToWorld[2][1] = down[2]
    camToWorld[0][2] = forward[0]
    camToWorld[1][2] = forward[1]
    camToWorld[2][2] = forward[2]

    camToWorld[0][3] = src[0]
    camToWorld[1][3] = src[1]
    camToWorld[2][3] = src[2]
    camToWorld[3][3] = 1.

    worldToCam = np.linalg.inv(camToWorld)

    return worldToCam[:3, :3], worldToCam[:3, 3:4]
