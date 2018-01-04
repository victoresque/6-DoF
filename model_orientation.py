from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, concatenate
from keras.engine.topology import Layer
import keras.backend as K
import numpy as np

def get_model(input_shape):
    input = Input(input_shape)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(input)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(3, activation='linear', name='output')(x)

    return Model(input, [output])

def my_mse(y_true, y_pred):
    diff = K.abs(y_pred - y_true)
    return K.mean(K.square(K.minimum(K.constant([1., 0., 1.]) - diff, diff)))

def my_rmse(y_true, y_pred):
    return K.sqrt(my_mse(y_true, y_pred))