import numpy as np
import h5py
import argparse
import contextlib
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, LSTM, Dense, Activation, Bidirectional
from tensorflow.keras.losses import *
import tensorflow.keras.backend as K
import sys
from scipy.stats import mode
from collections import defaultdict
from itertools import compress

parser = argparse.ArgumentParser()
parser.add_argument('--study', type=str, default='REAL',
                    help='study name',choices=['CIS','REAL'])
parser.add_argument('--symptom', type=str, default='tremor',
                    help='study name',choices=['medication','dyskinesia','tremor'])

def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename.encode(), fapl=propfaid)) as fid:
        return h5py.File(fid, mode)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def print_scores(sub_ids,sub_scores):
  for n in range(len(sub_ids)):
    print("\nScores for subject: {}\n".format(sub_ids[n]))

    for item in range(len(sub_scores[n])):
      print("Split {}: {}".format(item,np.array(sub_scores[n][item])))

def zero_pad(features,max_len=None):
    """
    zero pad examples to the right until max_len
    """
    shapes = [item.shape[0] for item in features]
    shape1 = features[0].shape[1]
    if max_len==None:
        max_len = max(shapes)
    pad_values = [max_len - item.shape[0] for item in features]
    for n in range(len(pad_values)):
        if pad_values[n]>0:
            zeros = np.zeros([pad_values[n],shape1])
            features[n] = np.concatenate((zeros,features[n]),axis=0)
    return features, max_len

def seq2one(vocab, input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(123,input_shape=input_shape)))
    model.add(Dense(vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy',precision_m,recall_m,f1_m])
    return model
    
def train_model(train_inputs, train_labels):
    epochs = 5
    batch_size = 16
    vocab = 5
    model = seq2one(vocab, input_shape=(max_len,train_inputs[0].shape[1]))
    model.fit(np.array(train_inputs),np.array(train_labels),batch_size=batch_size,verbose=1,epochs=epochs)
    return model

#-----------------------------------------------------------------------------------------------

args = parser.parse_args()
study = args.study
symptom = args.symptom

if study == "CIS":
    train_file="../Datasets/CIS/training_data/CIS_Train.hdf5"
    test_file="../Datasets/CIS/testing_data/CIS_Test.hdf5"

if study == "REAL":
    train_file = "../Datasets/REAL/training_data/smartwatch_accelerometer/REAL_Train.hdf5"
    test_file = "../Datasets/REAL/testing_data/smartwatch_accelerometer/REAL_Test.hdf5"

results = defaultdict(list)

train_f = hdf5_handler(train_file,'r')
test_f = hdf5_handler(test_file,'r')

subjects_list = [subject for subject in train_f.keys()]

for subject in subjects_list:
    
    print("Subject: ",subject)
    d_train = dict(train_f[subject]['measurements'])
    d_test = dict(test_f[subject]['measurements'])
    examples = [k for (k,v) in d_train.items() if 'time_series' in k]
    train_features = [v[:] for (k,v) in d_train.items() if 'time_series' in k]
    test_features = [v[:] for (k,v) in d_train.items() if 'time_series' in k]
    test_measurement_ids = np.stack([ids.decode('utf-8') for ids in test_f[str(subject)]['ids'][()]])
    
    # features, max_len = aug_pad(features)
    train_features, max_len = zero_pad(train_features)
    test_features, _ = zero_pad(test_features,max_len=max_len)

    if symptom == 'medication': label_index = 0
    if symptom == 'dyskinesia': label_index = 1
    if symptom == 'tremor': label_index = 2

    train_labels = [v[:][:][label_index] for v in train_f[str(subject)]['labels']]
    train_features = list(compress(train_features,np.stack(train_labels)>=0))
    train_labels = list(compress(train_labels,np.stack(train_labels)>=0))
    # if any(element<0 for element in labels):
    if len(train_labels)==0:
      print("NaN Values. Skipping subject...")
      continue
    train_labels = to_categorical(train_labels,5)

    # train_inputs, test_inputs, train_labels, test_labels = train_test_split(features, labels, train_size=0.8, random_state=42)
    model = train_model(train_features, train_labels)
    preds = np.argmax(model.predict(test_features))

    preds_csv = pd.DataFrame()
    preds_csv['measurement_ids'] = test_measurement_ids
    preds_csv['prediction'] = preds
    
    
    
    
    
    
    
    
    