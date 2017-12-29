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
from model_orientation import ab_rmse
from myutils.render import uv2ab

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
        u = view['u']
        v = view['v']
        a, b = uv2ab(u, v)
        views.append([a - 0.5, 2 * b / np.pi - 0.5])

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
    return K.sqrt(K.mean(K.square(K.minimum(K.constant([1., 0.]) - diff, diff)))) * 360

opt = keras.optimizers.adam()
model.compile(loss=ab_rmse,
              optimizer=opt,
              metrics=[deg_diff])

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=16.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.05,
    channel_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)
datagen.fit(x_train)

from keras.callbacks import ModelCheckpoint
callbacks = [ModelCheckpoint('ori_{epoch:03d}_{val_loss:.4f}.h5', period=5)]
model.fit_generator(datagen.flow(x_train, y_train, batch_size=256),
                    steps_per_epoch=256,
                    epochs=500,
                    callbacks=callbacks,
                    validation_data=(x_test, y_test))

model.save('ori.h5')
