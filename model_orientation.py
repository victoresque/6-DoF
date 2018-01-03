from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, concatenate
from keras.engine.topology import Layer
import keras.backend as K
import numpy as np

def get_model(input_shape):
    input = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = BatchNormalization()(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)
    pool3 = Flatten()(pool3)

    '''
    vp_pool2 = MaxPooling2D((3, 3))(conv2)
    vp_conv2 = Conv2D(128, (3, 3), activation='relu')(vp_pool2)
    vp_conv2 = Conv2D(128, (3, 3), activation='relu')(vp_conv2)
    vp_pool2 = MaxPooling2D((2, 2))(vp_conv2)
    vp_pool2 = Flatten()(vp_pool2)
    vp_dense1 = Dense(512, activation='relu')(vp_pool2)
    vp_dense1 = Dense(512, activation='relu')(vp_dense1)
    vp_dense1 = Dense(9, activation='softmax', name='vp')(vp_dense1)
    vp = vp_dense1
    pool3 = concatenate([pool3, vp_pool2])
    '''
    dense1 = Dense(512, activation='relu')(pool3)
    dense1 = Dense(512, activation='relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    output = Dense(3, activation='linear', name='output')(dense1)

    return Model(input, [output])

def my_mse(y_true, y_pred):
    diff = K.abs(y_pred - y_true)
    return K.mean(K.square(K.minimum(K.constant([2 * np.pi, 0., 2 * np.pi]) - diff, diff)))

def my_rmse(y_true, y_pred):
    return K.sqrt(my_mse(y_true, y_pred))