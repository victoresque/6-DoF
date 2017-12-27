from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.engine.topology import Layer

def get_model(num_classes, input_shape):
    input = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Flatten()(pool2)
    dense1 = Dense(512, activation='relu')(pool2)
    dense1 = Dropout(0.2)(dense1)
    dense1 = Dense(512, activation='relu')(dense1)
    dense1 = Dropout(0.2)(dense1)
    output = Dense(3, activation='linear')(dense1)
    return Model(input, output)
