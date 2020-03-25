import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import  MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import  Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform

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

def get_zhang_model(input_shape,num_classes):
    # Model architecture
    model = Sequential()
    model.add(Conv2D(128, (5,3), activation='relu', input_shape=input_shape, padding='same',
                kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(1,3)))
    # model.add(ZeroPadding2D(padding=(2,0)))
    # model.add(Conv2D(128, (5,3), activation='relu', kernel_initializer=initialize_weights, 
    #             bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    # model.add(ZeroPadding2D(padding=(2,0)))
    # model.add(Conv2D(128, (5,3), activation='relu', kernel_initializer=initialize_weights, 
    #             bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    # model.add(ZeroPadding2D(padding=(2,0)))
    # model.add(Conv2D(128, (5,3), activation='relu', kernel_initializer=initialize_weights, 
    #             bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(ZeroPadding2D(padding=(2,0)))
    model.add(Conv2D(128, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(ZeroPadding2D(padding=(2,0)))
    model.add(Conv2D(64, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(ZeroPadding2D(padding=(2,0)))
    model.add(Conv2D(32, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(ZeroPadding2D(padding=(2,0)))
    model.add(Conv2D(16, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(ZeroPadding2D(padding=(2,0)))
    model.add(Conv2D(8, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(ZeroPadding2D(padding=(2,0)))
    model.add(Conv2D(4, (5,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(1e-3), 
                kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(1e-3), 
                kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(1e-3), 
                kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(Dense(num_classes,activation='softmax'))
    
    return model

def get_dcnn_model(input_shape,num_classes):
    # Model architecture
    model = Sequential()
    model.add(Conv2D(128, (3,3), activation='relu', input_shape=input_shape, padding='same',
                kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(1,2)))
    model.add(Conv2D(256, (3,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(ZeroPadding2D(padding=(2,0)))
    model.add(Conv2D(256, (3,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(ZeroPadding2D(padding=(2,0)))
    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(1,2)))
    model.add(ZeroPadding2D(padding=(2,0)))
    model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(1,2)))
    model.add(ZeroPadding2D(padding=(2,0)))
    model.add(Conv2D(16, (3,3), activation='relu', kernel_initializer=initialize_weights, 
                bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(1e-3), 
                kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(1e-3), 
                kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(1e-3), 
                kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(Dense(num_classes,activation='softmax'))
    
    return model

def CNN_2D(input_shape):
    
    model = Sequential()
    model.add(Conv2D(64, (4,9), activation='relu', input_shape=input_shape, kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,10), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (5,5), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,6), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4),data_format='channels_first'))
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

def VGG16(input_shape,num_classes):

    model = Sequential()
    model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    return model