from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers import concatenate
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.engine.topology import Layer
import keras.backend as K
import numpy as np

def get_model(input_shape):
    img = Input(input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(img)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(1024, (1, 1), activation='relu')(x)
    x = Conv2DTranspose(1, kernel_size=(24, 24), strides=(12, 12), activation='sigmoid')(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(img, decoded)
