from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.engine.topology import Layer

def get_model(num_classes, input_shape):
    input = Input(input_shape)
    conv1_1 = Conv2D(16, (3, 3), activation='relu')(input)
    conv1_2 = Conv2D(16, (3, 3), activation='relu')(conv1_1)
    pool1 = MaxPooling2D((2, 2))(conv1_2)
    conv2_1 = Conv2D(32, (3, 3), activation='relu')(pool1)
    conv2_2 = Conv2D(32, (3, 3), activation='relu')(conv2_1)
    pool2 = MaxPooling2D((2, 2))(conv2_2)
    conv3_1 = Conv2D(64, (3, 3), activation='relu')(pool2)
    conv3_2 = Conv2D(64, (3, 3), activation='relu')(conv3_1)
    pool3 = MaxPooling2D((2, 2))(conv3_2)

    flatten_4 = Flatten()(pool3)
    fc_4 = Dense(128)(flatten_4)
    output = Dense(num_classes, activation='softmax')(fc_4)
    return Model(input, output)