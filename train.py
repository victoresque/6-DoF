import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from tqdm import tqdm
from data import getBackgrounds
from params import *
from model import *
from augment import *
from sample import *

bgs = getBackgrounds(300)
for model_id in model_ids:
    render_base = synth_base + 'orientation/{:02d}/render/'.format(model_id)
    udpstyle_base = synth_base + 'udpstyle/{:02d}/'.format(model_id)

    def getSyntheticData(path, with_info):
        images = []
        images_info = []
        filenames = sorted(os.listdir(path))[:100]
        for filename in tqdm(filenames, 'Reading synthetic'):
            ext = os.path.splitext(filename)[1]
            if ext == '.png':
                images.append(np.asarray(Image.open(path + filename)))
            if ext == '.json':
                images_info.append(json.load(open(path + filename, 'r')))
        if with_info:
            return images, images_info
        else:
            return images

    images, images_info = getSyntheticData(render_base, True)
    images_accv, images_info_accv = getSyntheticData(udpstyle_base, True)

    images = [cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA) for image in images]
    images = np.array(images)
    images = np.array([randomPaste(x_, bgs) for x_ in images])

    for img in images:
        cv2.imshow('', img)
        cv2.waitKey()

    images_accv = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images_accv]
    images_accv = np.array(images_accv)

    pivots = [[pivot for pivot in image_info['pivots']] for image_info in images_info]
    pivots = [np.array(pivot).flatten() for pivot in pivots]
    pivots = (np.array(pivots) - render_resize / 2) / render_resize

    pivots_accv = [[pivot for pivot in image_info['pivots']] for image_info in images_info_accv]
    pivots_accv = [np.array(pivot).flatten() for pivot in pivots_accv]
    pivots_accv = (np.array(pivots_accv) - render_resize / 2) / render_resize
    '''
    for i, image in enumerate(images):
        scale = np.random.uniform(0.6, 0.9)
        scaler = iaa.Affine(scale=scale)
        image = scaler.augment_image(image)
        image = randomPaste(image, bgs).copy()
        yy = pivots[i].reshape((-1, 2)) * scale
        for p in yy:
            u = int(p[0] * 96 + 48)
            v = int(p[1] * 96 + 48)
            if 0 <= u < 96 and 0 <= v < 96:
                image[v, u, :] = np.array([0.0, 0.0, 255.0])
        cv2.imshow('', image)
        cv2.waitKey()
    '''
    # seq.show_grid(cv2.cvtColor(images[12], cv2.COLOR_BGR2RGB), 8, 8)

    model = CNN()
    continue_from = 0
    # model.load_state_dict(torch.load('models/model_epoch001_loss_0.054722.pth'))
    model.cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Trainable parameters:', params)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)

    n_input = len(images)
    batch_size = 32
    def train(epoch, x, y):
        print('Epoch {:02d}:'.format(epoch))

        model.train()
        total_loss = 0.
        total_val_loss = 0.

        n_batch = n_input // batch_size
        val_split = 0.1

        for batch_id in range(n_batch):

            x_batch = x[batch_id * batch_size: (batch_id+1) * batch_size]

            scale = np.random.uniform(0.5, 1.8)
            scaler = iaa.Affine(scale=scale)
            x_batch = scaler.augment_images(x_batch)
            x_batch = np.array([randomPaste(x_, bgs) for x_ in x_batch])
            x_batch = seq.augment_images(x_batch)
            
            y_batch = y[batch_id * batch_size: (batch_id + 1) * batch_size].astype(np.float)
            y_batch = y_batch * scale

            images_a, pivots_a = shuffle(images_accv, pivots_accv)
            x_batch = np.append(x_batch, images_a[:batch_size], axis=0)
            y_batch = np.append(y_batch, pivots_a[:batch_size], axis=0)

            '''
            for i, img in enumerate(x_batch):
                yy = np.reshape(y_batch[i], (-1, 2))
                for p in yy:
                    u = int(p[0] * 96 + 48)
                    v = int(p[1] * 96 + 48)
                    if 0 <= u < 96 and 0 <= v < 96:
                        img[v][u] = np.array([0.0, 255.0, 255.0])
                cv2.imshow('o', img)
                cv2.waitKey()
            '''
            x_batch = x_batch / 255 - 0.5
            x_batch = np.transpose(x_batch, (0, 3, 1, 2))

            output = model(Variable(torch.FloatTensor(x_batch)).cuda())
            loss = criterion(output, Variable(torch.FloatTensor(y_batch)).cuda())

            # Validation
            if batch_id > n_batch * (1 - val_split):
                total_val_loss += loss.data[0]
                continue

            total_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % batch_size == 0:
                print('Epoch {:02d}: [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_id * len(x_batch), n_input,
                    100. * batch_id / (n_batch * (1 - val_split)), loss.data[0]))

        avg_loss = total_loss / (n_batch * (1 - val_split))
        print('Average loss: {:.6f}'.format(avg_loss))
        if val_split > 0:
            avg_val_loss =  total_val_loss / (n_batch * val_split)
            print('Average validation loss: {:.6f}'.format(avg_val_loss))
            torch.save(model.state_dict(),
                       'models/model_epoch{:03d}_loss_{:.6f}_val_{:.6f}.pth'.format(epoch, avg_loss, avg_val_loss))
        else:
            torch.save(model.state_dict(),
                       'models/model_epoch{:03d}_loss_{:.6f}.pth'.format(epoch, avg_loss))
        return train_loss

    train_loss = []
    images, pivots = shuffle(images, pivots)
    for epoch in range(continue_from + 1, 99999 + 1):
        train_loss.extend(train(epoch, images, pivots))
