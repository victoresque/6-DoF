import numpy as np
import cv2
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sixd.pysixd.view_sampler import fibonacci_sampling
from params import *


def drawBoundingBox(image, l, t, r, b):
    out = np.copy(image)
    lt = (int(l * image.shape[1]), int(t * image.shape[0]))
    rb = (int(r * image.shape[1]), int(b * image.shape[0]))
    cv2.rectangle(out, lt, rb, (0, 255, 0), 2)
    return out


def getBoundingBox(image):
    a = np.where(image != 0)
    return np.min(a[1])/640, np.min(a[0])/480, \
           np.max(a[1])/640, np.max(a[0])/480


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


def getRandomView(radius):
    theta = np.random.uniform(-np.pi, np.pi)
    u = np.random.uniform(-1, 1)
    v = np.random.uniform(-1, 1)
    x = np.sqrt(1 - np.square(u)) * np.cos(v * np.pi)
    y = np.sqrt(1 - np.square(u)) * np.sin(v * np.pi)
    z = u
    rvec = np.array([x, y, z])
    rvec = rvec / np.linalg.norm(rvec) * theta
    R = cv2.Rodrigues(rvec)[0]
    t = np.array([0, 0, radius])
    if R[2][2] > 0:
        return getRandomView(radius)
    else:
        return {'R': R, 't': t, 'rvec': rvec.tolist()}


def getViews(view_count, view_radius, inplane_steps, randomized=False, upper_only=True):
    if not upper_only:
        view_count = view_count // 2
    viewpoints = fibonacci_sampling(n_pts=view_count + (view_count + 1 % 2), radius=view_radius)
    viewpoints = np.array(viewpoints)
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    ax.scatter(viewpoints[:, 0], viewpoints[:, 1], viewpoints[:, 2])
    plt.show()
    '''
    views = []
    for i, vp in enumerate(tqdm(viewpoints, 'Generating views: ')):
        if vp[2] > -0.2 * view_radius:
            R, t = lookAt(vp, [0, 0, 0])
            for i in range(inplane_steps):
                rz = i * (np.pi * 2 / inplane_steps)
                if randomized:
                    rz = rz + np.random.uniform(-np.pi / inplane_steps, np.pi / inplane_steps)
                R_ = np.matmul([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], R)
                views.append({'R': R_, 't': t, 'vp': vp, 'rz': rz})
    return views


def getRandomViews(view_count, view_radius):
    views = []
    for i in tqdm(range(view_count), 'Generating random views: '):
        views.append(getRandomView(view_radius))
    return views


def getRandomLights(view_count):
    lights = []
    for i in range(view_count):
        lights.append([np.random.uniform(-light_shift, light_shift),
                       np.random.uniform(0, light_shift),
                       np.random.uniform(-light_shift, light_shift)])
    return lights


def getPivots(xmin, xmax, ymin, ymax, zmin, zmax, step, u0, v0, resize_ratio, K, R, t, shrink):
    pivots = []
    xmin, xmax = xmin + shrink * (xmax - xmin) / 2, xmax - shrink * (xmax - xmin) / 2
    ymin, ymax = ymin + shrink * (ymax - ymin) / 2, ymax - shrink * (ymax - ymin) / 2
    zmin, zmax = zmin + shrink * (zmax - zmin) / 2, zmax - shrink * (zmax - zmin) / 2
    for i in range(step):
        for j in range(step):
            for k in range(step):
                p_obj = np.array([[xmin + (xmax - xmin) * i / (step - 1)],
                                  [ymin + (ymax - ymin) * j / (step - 1)],
                                  [zmin + (zmax - zmin) * k / (step - 1)]])
                p_cam = np.matmul(R, p_obj) + t
                p_screen = (np.matmul(K, p_cam) / p_cam[2] - np.array([[u0], [v0], [0]])) * resize_ratio
                pivots.append([p_obj.squeeze().tolist(), p_screen.flatten()[:2].tolist()])
    return pivots


def rot2Angle(R):
    return np.arctan2(R[2][1], R[2][2]), \
           np.arctan2(-R[2][0], np.sqrt(R[2][1]**2 + R[2][2]**2)), \
           np.arctan2(R[1][0], R[0][0])


def angle2Rot(rx, ry, rz):
    X = [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    Y = [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    Z = [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    return np.matmul(Z, np.matmul(Y, X))



