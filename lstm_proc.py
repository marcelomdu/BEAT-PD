import numpy as np
import h5py 
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.text import one_hot
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, LSTM, Dense, Activation, Bidirectional
from keras.losses import *
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


if __name__ == '__main__':
    path = '/usr/share/datasets/BEAT-PD/' 
    subjects_list = ['1006','1032','1049']
    subject = subjects_list[0]

    file = path+"training_data_preclustering_subset.hdf5"
    f = h5py.File(file,'r')

    d = dict(f[subject]['measurements'])
    examples = [k for (k,v) in d.items() if 'wfeatures' in k]
    features =  [v[:] for (k,v) in d.items() if 'wfeatures' in k]
    # flabels =  [v[:] for (k,v) in d.items() if 'wlabels' in k]

    features, max_val = zero_pad(features)

    labels = [v[:][:][0] for v in f[subject]['labels']]
    labels = to_categorical(labels,5)

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(features, labels, train_size=0.8, random_state=42)

    ##get augmented dataset:
    # train_inputs_augmented = []
    # train_labels_augmented = []
    # for n in range(len(train_inputs)):
    #         augmented_example = []
    #         augmented_labels =[]
    #         augmented_input = generate_subsets(train_inputs[n],5)
    #         augmented_label = generate_subsets(train_labels[n],5)
    #         train_inputs_augmented.append(augmented_input)
    #         train_labels_augmented.append(augmented_label)
    # train_inputs = train_inputs_augmented
    # train_labels = train_labels_augmented

    vocab = 5
    model = seq2one(vocab, input_shape=(max_val,72))
    model.fit(np.array(train_inputs),np.array(train_labels),batch_size=16,verbose=1,epochs=15)

    score = model.evaluate(np.array(test_inputs), np.array(test_labels), verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test precision:', score[2])
    print('Test recall:', score[3])
    print('Test f1:', score[4])