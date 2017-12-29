import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy
import keras.backend as K
import os
import numpy as np
import cv2
import json
from tqdm import tqdm
import model_orientation
from model_orientation import abc_rmse

synth_base = '/home/victorhuang/Desktop/pose/algorithms/synthetic/orientation/'
img_base = synth_base + 'img/'
img_name = sorted(os.listdir(img_base))
view_base = synth_base + 'gt/'
view_name = sorted(os.listdir(view_base))

images = []
views = []
for filename in tqdm(img_name):
    images.append(cv2.imread(img_base + filename))
for filename in tqdm(view_name):
    with open(view_base + filename, 'r') as f:
        view = json.load(f)
        a, b, c = view['a'], view['b'], view['c']
        views.append([a / 2 / np.pi - 0.5,
                      b / (np.pi / 2) - 0.5,
                      c / (np.pi / 2)])

x_train = np.array(images)
y_train = np.array(views)

test_split = int(len(x_train) * 0.1)
x_test = x_train[:test_split]
y_test = y_train[:test_split]
x_train = x_train[test_split:]
y_train = y_train[test_split:]

model = model_orientation.get_model(x_train.shape[1:])
model.summary()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def deg_diff(y_true, y_pred):
    diff = K.abs(y_pred - y_true)
    return K.sqrt(K.mean(K.square(K.minimum(K.constant([1., 0., 0.]) - diff, diff)))) * 360

opt = keras.optimizers.adam(decay=0.02)
model.compile(loss=abc_rmse,
              optimizer=opt,
              metrics=[deg_diff])

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=5.,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    shear_range=0.,
    channel_shift_range=0.05,
    horizontal_flip=False,
    vertical_flip=False)
datagen.fit(x_train)

from keras.callbacks import ModelCheckpoint
callbacks = [ModelCheckpoint('ori_{epoch:03d}_{val_loss:.4f}.h5', period=5)]
model.fit_generator(datagen.flow(x_train, y_train, batch_size=512),
                    steps_per_epoch=128,
                    epochs=500,
                    callbacks=callbacks,
                    validation_data=(x_test, y_test))
