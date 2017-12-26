import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import cv2
import json
from tqdm import tqdm
import model_orientation

synth_base = '/home/victorhuang/Desktop/pose/algorithms/Pose6D/synthetic/orientation/'
img_base = synth_base + 'img/'
img_name = os.listdir(img_base)
view_base = synth_base + 'gt/'
view_name = os.listdir(view_base)

images = []
views = []
for filename in tqdm(img_name):
    images.append(cv2.imread(img_base + filename))
for filename in tqdm(view_name):
    with open(view_base + filename, 'r') as f:
        views.append(json.load(f)['view'])

num_classes = 8
epochs = 200

for i in range(len(images)):
    images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
x_train = np.array(images)
x_train = np.expand_dims(x_train, axis=3)
y_train = np.array(views)

test_split = int(len(x_train) * 0.1)
x_test = x_train[:test_split]
y_test = y_train[:test_split]
x_train = x_train[test_split:]
y_train = y_train[test_split:]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = model_orientation.get_model(num_classes, x_train.shape[1:])
model.summary()

opt = keras.optimizers.adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train,
          batch_size=512,
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose=2)

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
    shear_range=0.025,
    channel_shift_range=0.025,
    horizontal_flip=False,
    vertical_flip=False)
datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=256),
                    steps_per_epoch=256,
                    epochs=epochs,
                    validation_data=(x_test, y_test))
