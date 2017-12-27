import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy
import os
import numpy as np
import cv2
import json
from tqdm import tqdm
import model_orientation

synth_base = '/home/victorhuang/Desktop/pose/algorithms/synthetic/orientation/'
img_base = synth_base + 'img/'
img_name = sorted(os.listdir(img_base))
view_base = synth_base + 'gt/'
view_name = sorted(os.listdir(view_base))
num_classes = 9

images = []
views = []
for filename in tqdm(img_name):
    images.append(cv2.imread(img_base + filename))
for filename in tqdm(view_name):
    with open(view_base + filename, 'r') as f:
        view = json.load(f)
        views.append([view['rx'], view['ry'], view['rz']])

x_train = np.array(images)
y_train = np.array(views)

test_split = int(len(x_train) * 0.1)
x_test = x_train[:test_split]
y_test = y_train[:test_split]
x_train = x_train[test_split:]
y_train = y_train[test_split:]

# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = model_orientation.get_model(num_classes, x_train.shape[1:])
model.summary()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def top_2_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

opt = keras.optimizers.adam()
model.compile(loss='mse',
              optimizer=opt)
'''
model.fit(x_train, y_train, batch_size=32,
          epochs=100, validation_data=(x_test, y_test))
'''
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=8.,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.005,
    channel_shift_range=0.005,
    horizontal_flip=False,
    vertical_flip=False)
datagen.fit(x_train)

from keras.callbacks import ModelCheckpoint
callbacks = [ModelCheckpoint('ori_{epoch:03d}_{val_loss:.4f}.h5', period=5)]
model.fit_generator(datagen.flow(x_train, y_train, batch_size=256),
                    steps_per_epoch=128,
                    epochs=500,
                    callbacks=callbacks,
                    validation_data=(x_test, y_test))

model.save('ori.h5')
