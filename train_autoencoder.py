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
nobg_base = synth_base + 'nobg/{:02d}/'.format(model_id)
nobg_name = sorted(os.listdir(nobg_base))[:img_count]

images = []
seg = []
for filename in tqdm(img_name):
    image = cv2.imread(img_base + filename)
    images.append(image)
for filename in tqdm(nobg_name):
    image = cv2.imread(nobg_base + filename)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = (image > 0.).astype(np.float32)
    seg.append(image)

seq_affine = iaa.Sequential([
    iaa.Affine(translate_percent=(-0.1, 0.1),
               scale=(0.9, 1.1),
               rotate=(-2, 2),
               shear=(-2, 2),
               mode='edge')
])
seq_color = iaa.Sequential([
    #iaa.OneOf([
    #    iaa.GaussianBlur(sigma=(0, 1.0)),
    #    iaa.Sequential([
    #        iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.1),
    #        iaa.Affine(scale=(1.05, 1.05))
    #    ])
    #]),
    iaa.GaussianBlur(sigma=(1.25, 1.75)),
    # iaa.Add((-5, 5), per_channel=0.2),
    iaa.Add((8, 32)),
    iaa.Multiply((0.95, 1.05), per_channel=0.25),
    # iaa.Multiply((0.9, 1.1)),
    iaa.ContrastNormalization((0.9, 1.1))
])
seq = iaa.Sequential([
    seq_affine,
    seq_color
])
# seq.show_grid(cv2.cvtColor(images[4], cv2.COLOR_RGB2BGR), 8, 8)

x_train = np.array(images)
x_train = x_train.astype(np.float32) / 255
y_train = np.array(seg)
y_train = np.expand_dims(y_train, 3)

test_split = int(len(x_train) * 0.1)
x_test = x_train[:test_split]
y_test = y_train[:test_split]
x_train = x_train[test_split:]
y_train = y_train[test_split:]

import keras
import keras.backend as K
from keras.utils import to_categorical
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from model_autoencoder import get_model

model = get_model(x_train.shape[1:])
model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model, 'model.png')

opt = keras.optimizers.adam(decay=0.02)
model.compile(loss='mse',
              optimizer=opt)

from keras.callbacks import ModelCheckpoint
callbacks = [ModelCheckpoint('ori_{epoch:03d}_{loss:.4f}_{val_loss:.4f}.h5', period=5)]

class DataGenerator(object):
    def __init__(self, batch_size = 32):
      self.batch_size = batch_size

    def flow(self, x, y):
        while 1:
            indexes = np.arange(len(x))
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                list_IDs_temp = indexes[i*self.batch_size:(i+1)*self.batch_size]
                x_batch, y_batch = self.__data_generation(x, y, list_IDs_temp)
                yield x_batch, y_batch

    def __data_generation(self, x, y, list_IDs_temp):
        x_batch = np.empty((self.batch_size, 96, 96, 3))
        y_batch = np.empty((self.batch_size, 96, 96, 1))
        for i, ID in enumerate(list_IDs_temp):
            seq_affine_det = seq_affine.to_deterministic()
            # seq_color_det = seq_color.to_deterministic()
            x_batch[i] = seq_affine_det.augment_image(seq_color.augment_image(x[ID]))
            y_batch[i] = seq_affine_det.augment_image(y[ID])
        return x_batch, x_batch

batch_size = 64
datagen = DataGenerator(batch_size=batch_size)
valdatagen = DataGenerator(batch_size=batch_size)
model.fit_generator(generator=datagen.flow(x_train, y_train),
                    steps_per_epoch=len(x_train)//batch_size,
                    validation_data=valdatagen.flow(x_train, y_train),
                    validation_steps=len(x_test)//batch_size//2,
                    epochs=500, callbacks=callbacks)
