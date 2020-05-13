import numpy as np
import h5py 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, LSTM, Dense, Activation, Bidirectional
from tensorflow.keras.losses import *
import tensorflow.keras.backend as K

from plot_cm import plot_confusion_matrix

import sys
import h5py
from scipy.stats import mode

def generate_subsets(data, _min):
    """
    generates list of augmented examples
    """
    result = []
    for x in range(0,len(data) - _min + 1):
        for y in range(0, len(data) -_min +1):
            if len(data[x: len(data) - y]) < _min:
                break
            result.append(data[x: len(data) - y])
    # result.remove(data)
    return result
   
def seq2one(vocab, input_shape):
    
    model = Sequential()

    model.add(Bidirectional(LSTM(1024,input_shape=input_shape)))
    model.add(Dense(vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy',precision_m,recall_m,f1_m])
    return model

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


def zero_pad(features):
    """
    zero pad examples to the right until max_len
    """
    shapes = [item.shape[0] for item in features]
    shape1 = features[0].shape[1]
    max_val = max(shapes)
    pad_values = [max_val - item.shape[0] for item in features]
    for n in range(len(pad_values)):
        if pad_values[n]>0:
            zeros = np.zeros([pad_values[n],shape1])
            features[n] = np.concatenate((zeros,features[n]),axis=0)
    
    return features, max_val

def aug_pad(ft):
    """
    zero pad examples to the right until max_len
    """
    shapes = [item.shape[0] for item in ft]
    shape1 = ft[0].shape[1]
    max_len = max(shapes)
    for i in range(0,len(ft)):
        s_len = ft[i].shape[0]
        if s_len < max_len:
            for j in range(0,np.min([int(np.round(max_len/s_len)),10])):
                ft[i] = np.vstack((ft[i],ft[i]))
            ft[i] = ft[i][-max_len:]
    
    return ft, max_len

def get_tremor_windows(ft,wlabels):
    tft = list()
    for i in range(0,len(ft)):
        tft.append(ft[i][np.where(wlabels[i]==1),:][0,:,:])
    
    return tft


if __name__ == '__main__':
    path = '/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/' 
    subjects_list = ['1004']
    subject = subjects_list[0]

    file = path+"training_data_lstm.hdf5"
    f = h5py.File(file,'r')

    d = dict(f[subject]['measurements'])
    examples = [k for (k,v) in d.items() if 'time_series' in k]
    features =  [v[:][:,-11:] for (k,v) in d.items() if 'time_series' in k]
    
    features, max_len = aug_pad(features)

    labels = [v[:][:][2] for v in f[subject]['labels']]
    labels = to_categorical(labels,5)

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(features, labels, train_size=0.8, random_state=42)

    vocab = 5
    model = seq2one(vocab, input_shape=(max_len,features[0].shape[1]))
    model.fit(np.array(train_inputs),np.array(train_labels),batch_size=1,verbose=1,epochs=15)

    score = model.evaluate(np.array(test_inputs), np.array(test_labels), verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test precision:', score[2])
    print('Test recall:', score[3])
    print('Test f1:', score[4])
    
    true = np.argmax(test_labels,axis=1)
    preds = np.argmax(model.predict(np.array(test_inputs)),axis=1)
    
    class_names = np.array([0,1,2,3])
    
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(true, preds, classes=class_names,
                        title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(true, preds, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')
