import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Conv3D, ZeroPadding3D, Activation, Input, concatenate
from tensorflow.keras.models import Model

# from keras.layers.normalization import BatchNormalization
# from keras.layers.pooling import  MaxPooling2D
# from keras.layers.merge import  Concatenate
# from keras.layers.core import Lambda, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import  MaxPooling2D, MaxPooling3D
from tensorflow.keras.layers import  Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform

# from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def initialize_weights(shape, dtype=None):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)

def initialize_bias(shape, dtype=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)

def get_siamese_model(input_shape,type='2D'):
    # Input tensors
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    if type=='2D':
        model = CNN_2D(input_shape)
    if type=='1D':
        model = CNN_1D(input_shape)
    
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

def get_zhang_model(input_shape):
    # Model architecture    
    model = Sequential()
    model.add(ZeroPadding3D(padding=(0,1,0)))
    model.add(Conv3D(128, (3,5,3), activation='relu', input_shape=input_shape, 
                kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Conv2D(256, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(256, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(256, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(256, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(256, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(256, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(256, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(256, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(256, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(1e-3), 
                kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(1e-3), 
                kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(1e-3), 
                kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(Dense(5,activatio='softmax'))
    
    return model

def CNN_2D(input_shape):
    
    model = Sequential()
    model.add(Conv2D(64, (4,9), activation='relu', input_shape=input_shape, kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,10), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (5,5), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,6), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3), kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

    return model


def CNN_1D(input_shape):
    
    model = Sequential()
    model.add(Conv1D(64, 9, activation='relu', input_shape=input_shape, kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 10, activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 5, activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 6, activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(1e-3), kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

    return model
