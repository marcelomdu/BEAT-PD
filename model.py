import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model

# from tensorflow.keras.layers.normalization import BatchNormalization
# from tensorflow.keras.layers.pooling import  MaxPooling2D
# from tensorflow.keras.layers.merge import  Concatenate
# from tensorflow.keras.layers.core import Lambda, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import  MaxPooling2D
from tensorflow.keras.layers import  Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform

# from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def initialize_weights(shape, dtype=None):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)

def initialize_bias(shape, dtype=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)

def get_siamese_model(input_shape):
    # Input tensors
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # CNN
    model = Sequential()
    # model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape, kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(128, (7,7), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    model.add(Conv2D(10, (4,4), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(20, (4,4), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3), kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

    # Feature vectors
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Calculate encoding distances
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0]-tensors[1]))
    L1_distance = L1_layer([encoded_l,encoded_r])

    # Calculate similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)

    # Connecti the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    return siamese_net
