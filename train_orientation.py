import os
import numpy as np
import cv2
import json
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm
from myutils.transform import rot2Angle

model_id = 6
img_count = 10000
synth_base = '/home/victorhuang/Desktop/pose/algorithms/synthetic/orientation/'
img_base = synth_base + 'img/{:02d}/'.format(model_id)
img_name = sorted(os.listdir(img_base))[:img_count]
view_base = synth_base + 'gt/{:02d}/'.format(model_id)
view_name = sorted(os.listdir(view_base))[:img_count]

images = []
views = []
Rs = []
for filename in tqdm(img_name):
    image = cv2.imread(img_base + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))
    images.append(image)
for filename in tqdm(view_name):
    with open(view_base + filename, 'r') as f:
        view = json.load(f)
        R = view['R']
        Rs.append(R)
        rx, ry, rz = rot2Angle(R)
        views.append([rx, ry, rz])

seq = iaa.Sequential([
    iaa.Affine(scale=(0.9, 1.1),
               translate_percent=(-0.025, 0.025),
               rotate=(-1, 1),
               shear=(-2, 2),
               mode='edge'),
    iaa.GaussianBlur(sigma=(0, 0.5)),
    iaa.Add((-5, 5), per_channel=0.2),
    iaa.Add((-32, 32)),
    iaa.Multiply((0.95, 1.05), per_channel=0.2),
    iaa.Multiply((0.9, 1.1)),
    iaa.ContrastNormalization((0.9, 1.1))
])
# seq.show_grid(images[0], cols=8, rows=8)

import keras
import keras.backend as K
from keras.utils import to_categorical
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
import model_orientation
from model_orientation import my_mse, my_rmse

x_train = np.array(images)
y_train = np.array(views)

y1 = []
t = 1 / np.sqrt(3)
viewpoints = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
                       [t, t, t], [t, -t, t], [-t, t, t], [-t, -t, t],
                       [0, 0, 1]])
for R in Rs:
    max_sim = 0
    vp_sim = None
    for i, vp in enumerate(viewpoints):
        Rvp = np.matmul(R, vp).flatten()
        sim = np.dot(Rvp, np.array([0, 0, -1]))
        if sim > max_sim:
            max_sim = sim
            vp_sim = i
    y1.append(to_categorical(vp_sim, num_classes=len(viewpoints)).squeeze())

y1 = np.array(y1)

test_split = int(len(x_train) * 0.1)
x_test = x_train[:test_split]
y_test = y_train[:test_split]
y1_test = y1[:test_split]
x_train = x_train[test_split:]
y_train = y_train[test_split:]
y1_train = y1[test_split:]

model = model_orientation.get_model(x_train.shape[1:])
model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model, 'model.png')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

def deg_diff(y_true, y_pred):
    return my_rmse(y_true, y_pred) * 360 / 2 / np.pi

opt = keras.optimizers.adam(decay=0.02)
model.compile(loss=['mse'],
              optimizer=opt,
              metrics=[deg_diff])

from keras.callbacks import ModelCheckpoint
callbacks = [ModelCheckpoint('ori_{epoch:03d}_{loss:.4f}_{val_loss:.4f}.h5', period=5)]

class DataGenerator(object):
    def __init__(self, batch_size = 32):
      self.batch_size = batch_size

    def flow(self, x, y, y1):
        while 1:
            indexes = np.arange(len(x))
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                list_IDs_temp = indexes[i*self.batch_size:(i+1)*self.batch_size]
                x_batch, y_batch, y1_batch = self.__data_generation(x, y, y1, list_IDs_temp)
                # yield x_batch, [y_batch, y1_batch]
                yield x_batch, [y_batch]

    def __data_generation(self, x, y, y1, list_IDs_temp):
        x_batch = np.empty((self.batch_size, 48, 48, 3))
        y_batch = np.empty((self.batch_size, 3), dtype=np.float32)
        y1_batch = np.empty((self.batch_size, 9), dtype=np.float32)
        for i, ID in enumerate(list_IDs_temp):
            x_batch[i] = seq.augment_image(x[ID])
            y_batch[i] = y[ID]
            y1_batch[i] = y1[ID]
        return x_batch, y_batch, y1_batch

batch_size = 128
datagen = DataGenerator(batch_size=batch_size)
valdatagen = DataGenerator(batch_size=batch_size)
model.fit_generator(generator=datagen.flow(x_train, y_train, y1_train),
                    steps_per_epoch=2*len(x_train)//batch_size,
                    validation_data=valdatagen.flow(x_train, y_train, y1_train),
                    validation_steps=len(x_test)//batch_size//2,
                    epochs=500, callbacks=callbacks)
