from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.engine.topology import Layer
import keras.backend as K

def get_model(input_shape):
    input = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Flatten()(pool3)
    dense1 = Dense(1024, activation='relu')(pool3)
    dense1 = Dropout(0.5)(dense1)
    dense1 = Dense(1024, activation='relu')(dense1)
    dense1 = Dropout(0.5)(dense1)
    output = Dense(2, activation='linear')(dense1)
    return Model(input, output)

def ab_rmse(y_true, y_pred):
    diff = K.abs(y_pred - y_true)
    return K.sqrt(K.mean(K.square(K.minimum(K.constant([1., 0.]) - diff, diff))))