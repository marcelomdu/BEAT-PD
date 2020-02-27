import sys
import numpy as np
import pandas as pd
import pickle
import os

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import  MaxPooling2D
from keras.layers.merge import  Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K

from sklearn.utils import shuffle

import numpy.random as npr


def initialize_weights(shape, name=None):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)

def initialize_bias(shape, name=None):
    return np.ranom.normal(loc=0.5, scale=1e-2, size=shape)

def get_siamese_model(input_shape):
    # Input tensors
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # CNN
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape, kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoit', kernel_regularizer=l2(1e-3), kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

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


def make_oneshot_task(val_data,val_labels,N):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    _, w, h = val_data.shape
    
    test_label = randint(0,5)
    
    # Matched and unmatched candidates
    m_candidates = val_data[val_labels==test_label,:,:]
    u_candidates = val_data[val_labels!=test_label,:,:]
    
    # Random indices for sampling
    m_idx1, m_idx2 = choice(m_candidates.shape[0]+1,replace=False,size=(2,)) # Non repetitive random indices
    u_indices = randint(0,u_candidates.shape[0]+1,size=(N,))
    
    # Matched image from support_set will be allocated to position '0' then shuffled
    test_image = np.asarray([m_candidates[m_idx1,:,:]]*N).reshape(N, w, h, 1)
    support_set = u_candidates[u_indices,:,:]
    support_set[0,:,:] = m_candidates[m_idx2,:,:]
    support_set = support_set.reshape(N, w, h, 1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)

    pairs = [test_image, support_set]

    return pairs, targets

  
  def test_oneshot(model,val_data,val_labels,N,k, verbose = 0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
    return percent_correct
