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

bgs = getBackgrounds(100)
for model_id in model_ids:
    render_base = synth_base + 'orientation/{:02d}/render/'.format(model_id)
    patch_base = synth_base + 'orientation/{:02d}/patch/'.format(model_id)

    def getSyntheticData(path, with_info):
        images = []
        images_info = []
        filenames = sorted(os.listdir(path))
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

    # images, images_info = getSyntheticData(render_base, True)
    images = getSyntheticData(patch_base, False)

    images = [cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA) for image in images]
    images = np.array(images)
    '''
    for image in images:
        cv2.imshow('', image)
        cv2.waitKey()
    '''
    # pivots = [[pivot[1] for pivot in image_info['pivots']] for image_info in images_info]
    # pivots = (np.array(pivots) - render_resize / 2) / render_resize

    # seq_color.show_grid(cv2.cvtColor(images[12], cv2.COLOR_BGR2RGB), 6, 6)

    model = ConvolutionalAutoEncoder()
    model.load_state_dict(torch.load('models/model_epoch87.pth'))
    model.cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Trainable parameters:', params)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_input = len(images)
    batch_size = 32
    def train(epoch, x):
        print('Epoch {:02d}:'.format(epoch))

        model.train()
        total_loss = 0.
        total_val_loss = 0.

        n_batch = n_input // batch_size
        val_split = 0.1

        for batch_id in range(n_batch):
            x_ = x[batch_id * batch_size: (batch_id+1) * batch_size].astype(np.float)
            x_ = np.array([randomPaste(_, bgs) for _ in x_])

            x_ = seq.augment_images(x_)
            # x_ = np.array([_.flatten() for _ in x_])
            x_ = x_ / 255
            # y_ = y[batch_id * batch_size: (batch_id+1) * batch_size].astype(np.float)
            # y_ = [_.flatten() for _ in y_]

            x_ = np.transpose(x_, (0, 3, 1, 2))
            output = model(Variable(torch.FloatTensor(x_)).cuda())
            # loss = criterion(output, Variable(torch.FloatTensor(y_)).cuda())
            loss = criterion(output, Variable(torch.FloatTensor(x_)).cuda())

            if batch_id > n_batch * (1 - val_split):
                total_val_loss += loss.data[0]
                continue

            total_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % batch_size == 0:
                print('Epoch {:02d}: [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_id * len(x_), n_input,
                    100. * batch_id / (n_batch * (1 - val_split)), loss.data[0]))

        avg_loss = total_loss / (n_batch * (1 - val_split))
        avg_val_loss =  total_val_loss / (n_batch * val_split)
        print('Average loss: {:.6f}'.format(avg_loss))
        print('Average validation loss: {:.6f}'.format(avg_val_loss))
        torch.save(model.state_dict(), 'models/model_epoch{}.pth'.format(epoch))
        return train_loss

    train_loss = []
    for epoch in range(1, 2000+1):
        train_loss.extend(train(epoch, images))
