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
from params import *
from model import *
from augment import *

for model_id in model_ids:
    patch_base = synth_base + 'orientation/{:02d}/patch/'.format(model_id)
    rand_patch_base = synth_base + 'orientation/{:02d}/rand_patch/'.format(model_id)

    def getSyntheticData(path, with_info):
        patches = []
        patches_info = []
        filenames = sorted(os.listdir(path))
        for filename in tqdm(filenames, 'Reading synthetic'):
            ext = os.path.splitext(filename)[1]
            if ext == '.png':
                patches.append(np.asarray(Image.open(path + filename)))
            if ext == '.json':
                patches_info.append(json.load(open(path + filename, 'r')))
        if with_info:
            return patches, patches_info
        else:
            return patches

    patches, patches_info = getSyntheticData(patch_base, True)
    rand_patches = getSyntheticData(rand_patch_base, False)

    patches = np.array(patches)
    rand_patches = np.array(rand_patches)

    # seq.show_grid(patches[123], cols=8, rows=8)

    def createPairs():
        x0 = []
        x1 = []
        label = []

        # True
        for p in tqdm(patches):
            x0.append(p)
            x1.append(p)
            label.append(1.)

        for i, p in tqdm(enumerate(patches)):
            if 1 <= i < len(patches) - 1:
                cor = np.array(patches_info[i]['cor'])
                cor_prev = np.array(patches_info[i - 1]['cor'])
                cor_next = np.array(patches_info[i + 1]['cor'])
                if np.linalg.norm(cor - cor_prev) < np.linalg.norm(cor - cor_next):
                    x0.append(p)
                    x1.append(patches[i - 1])
                else:
                    x0.append(p)
                    x1.append(patches[i + 1])
                label.append(1.)

        # False
        for p, rp in tqdm(zip(patches, rand_patches)):
            x0.append(p)
            x1.append(rp)
            label.append(0.)

        for i, p in tqdm(enumerate(patches)):
            shift = 12
            if shift <= i < len(patches) - shift - 1:
                im_id = np.array(patches_info[i]['img_id'])
                im_id_prev = np.array(patches_info[i - shift]['img_id'])
                im_id_next = np.array(patches_info[i + shift]['img_id'])
                if im_id == im_id_prev == im_id_next:
                    if np.random.rand() > 0.5:
                        x1.append(patches[i - shift])
                    else:
                        x1.append(patches[i + shift])
                elif im_id == im_id_prev:
                    x1.append(patches[i - shift])
                elif im_id == im_id_next:
                    x1.append(patches[i + shift])
                else:
                    break
                x0.append(p)
                label.append(0.)

        return x0, x1, label

    x0, x1, y = createPairs()
    x0 = np.array(x0)
    x1 = np.array(x1)
    y = np.array(y)
    print('False labels:', np.count_nonzero(y == 0))
    print('True labels:', np.count_nonzero(y == 1))

    model = SiameseNetwork()
    model.cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Trainable parameters:', params)

    lr = 1e-4
    momentum = 0.9
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    n_input = len(x0)
    batch_size = 64
    def train(epoch, x0_, x1_, y_):
        print('Epoch {:02d}:'.format(epoch))

        model.train()
        total_loss = 0.
        total_val_loss = 0.

        n_batch = n_input // batch_size
        val_split = 0.1

        for batch_id in range(n_batch):
            x0 = x0_[batch_id * batch_size: (batch_id+1) * batch_size].astype(np.float)
            x1 = x1_[batch_id * batch_size: (batch_id+1) * batch_size].astype(np.float)
            y = np.expand_dims(y_[batch_id * batch_size: (batch_id+1) * batch_size], 1)

            # Image augmentation
            x0 = seq.augment_images(x0)
            x1 = seq.augment_images(x1)

            x0 = np.transpose(x0, (0, 3, 1, 2))
            x1 = np.transpose(x1, (0, 3, 1, 2))

            x0 = torch.FloatTensor(x0)
            x1 = torch.FloatTensor(x1)
            y = torch.FloatTensor(y)

            x0 = Variable(x0).cuda()
            x1 = Variable(x1).cuda()
            y = Variable(y).cuda()

            output1, output2 = model(x0, x1)
            loss = criterion(output1, output2, y)

            if batch_id > n_batch * (1 - val_split):
                total_val_loss += loss.data[0]
                continue

            total_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % batch_size == 0:
                print('Epoch {:02d}: [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_id * len(y), n_input,
                    100. * batch_id / (n_batch * (1 - val_split)), loss.data[0]))

        avg_loss = total_loss / (n_batch * (1 - val_split))
        avg_val_loss =  total_val_loss / (n_batch * val_split)
        print('Average loss: {:.6f}'.format(avg_loss))
        print('Average validation loss: {:.6f}'.format(avg_val_loss))
        torch.save(model.state_dict(), 'models/model_epoch{}.pth'.format(epoch))
        return train_loss


    x0, x1, y = shuffle(x0, x1, y)

    train_loss = []
    for epoch in range(1, 200+1):
        train_loss.extend(train(epoch, x0, x1, y))
