import numpy as np
import h5py
import contextlib
import copy
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from numpy.random import randint
from itertools import compress
from tensorflow.keras.utils import to_categorical


def threshold_data_2D(data,labels,threshold):
    valid_data = list()
    valid_labels = list()
    for i in range(0,len(data)):
        if (data[i].shape[0] >= threshold):
            valid_data.append(data[i][:threshold,:])
            valid_labels.append(labels[i])
    return valid_data, valid_labels

def threshold_data_3D(data,labels,threshold):
    valid_data = list()
    valid_labels = list()
    for i in range(0,len(data)):
        if (data[i].shape[0] >= threshold):
            valid_data.append(data[i][-threshold:,:,:])
            valid_labels.append(labels[i])
    return valid_data, valid_labels

def get_train_test(data, labels, dim='3D', categorical=True, classes=[], n_tests=0, num_classes=5, threshold=True, th_value=100, balance=False):
    if threshold:
        if dim=='2D':
            valid_data, valid_labels = threshold_data_2D(data,labels,th_value)
        if dim=='3D':
            valid_data, valid_labels = threshold_data_3D(data,labels,th_value)
    else:
        valid_data = data
        valid_labels = labels
    
    if balance:
        num_classes = len(classes)
        X_train, X_test, y_train, y_test, _ = get_balanced_data(valid_data,valid_labels,classes,n_tests)
    else:
        X_train, X_test, y_train, y_test = train_test_split(valid_data, valid_labels, test_size=0.25)
        X_train = np.stack(X_train)
    
    X_test = np.stack(X_test)

    if categorical:
        if type(y_train) == dict:
            X_train, y_train = dict_to_array(X_train, y_train)
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

    return X_train, X_test, y_train, y_test

def get_pairs(data_train,labels_train):
    X = copy.deepcopy(data_train)
    l_matched = list()
    r_matched = list()
    l_unmatched = list()
    r_unmatched = list()
    n = int(len(X)/4)
    x_len = list()
    n_classes = len(X)
    
    for i in range(0,n_classes):
        for _ in range(0,int(len(X[i])/4)):
            k = randint(0,len(X[i]))
            l_matched.append(X[i].pop(k))
            k = randint(0,len(X[i]))
            r_matched.append(X[i].pop(k))
        x_len.append(len(X[i]))
        
    for i in range(0,n_classes-1):
        if np.sum(x_len)>1:
            l_choose = list(X.keys())
            n = np.argmin(x_len)
            l = l_choose[n]
            l_choose.pop(n)
            x_len.pop(n)
            for _ in range(0,len(X[l])):
                if len(l_choose)>0:
                    k1 = randint(0,len(X[l]))
                    l_unmatched.append(X[l].pop(k1))
                    k2 = randint(0,len(l_choose))
                    l2 = l_choose[int(k2)]
                if (l2 in list(X.keys())):
                    if len(X[l2]) > 0:
                        k1 = randint(0,len(X[l2]))
                        r_unmatched.append(X[l2].pop(k1))
                        x_len[int(k2)] -= 1
                if (l2 in list(X.keys())):
                    if len(X[l2])==0:
                        l_choose.pop(k2)
                        X.pop(l2)
                        x_len.pop(k2)
            X.pop(l)
            
    targets = np.hstack((np.ones(len(l_matched)),np.zeros(len(l_unmatched))))
    l_matched = np.stack(l_matched)
    l_unmatched = np.stack(l_unmatched)
    l_pairs = np.vstack((l_matched,l_unmatched))
    r_matched = np.stack(r_matched)
    r_unmatched = np.stack(r_unmatched)
    r_pairs = np.vstack((r_matched,r_unmatched))
    l_pairs = l_pairs.reshape(l_pairs.shape[0],l_pairs.shape[1],l_pairs.shape[2],l_pairs.shape[3])
    r_pairs = r_pairs.reshape(r_pairs.shape[0],r_pairs.shape[1],r_pairs.shape[2],r_pairs.shape[3])
    pairs = [l_pairs,r_pairs]
    
    return pairs, targets

def get_balanced_data(data,labels,classes,n_test):
    
    X_train = dict()
    X_test = list()
    y_train = dict()
    y_test = list()
    
    cat_data = dict()
    for i in classes:
        cat_data[i] = list()
        X_train[i] = list()
        y_train[i] = list()
    for i in range(0,len(labels)):
        if labels[i] in classes:
            cat_data[labels[i]].append(data[i])
    
    for i in classes:
        for _ in range(0, n_test):
            k = randint(0,len(cat_data[i]))
            X = cat_data[i].pop(k)
            X_test.append(X)
            y_test.append(i)
        for _ in range(0, min(len(cat_data[i]),100*n_test)):
            k = randint(0,len(cat_data[i]))
            X = cat_data[i].pop(k)
            X_train[i].append(X)
            y_train[i].append(i)


    return X_train, X_test, y_train, y_test, cat_data       

def dict_to_array(X,y):
    X_list = list()
    y_list = list()
    for i in X.keys():
        X_list.append(np.stack(X[i]))
    for i in y.keys():
        y_list.append(np.stack(y[i]))
    X_array = X_list[0]
    y_array = y_list[0]
    for i in range(1,len(X_list)):
        X_array = np.vstack((X_array,X_list[i]))
    for i in range(1,len(X_list)):
        y_array = np.hstack((y_array,y_list[i]))

    return X_array, y_array

def test_siamese(model, val_data, val_labels):
    
    val_data, val_labels = shuffle(val_data,val_labels)
    n_corrects = 0
    val_truth = list()
    val_pred = list()

    for i in range(0,len(val_labels)):
        val_idx = np.array(np.zeros(len(val_labels)),dtype=bool)
        val_idx[i] = True
        val_idx = np.invert(val_idx)
        pred_labels = list(np.compress(val_idx,val_labels))
        val_inputs = [(np.asarray([val_data[i,:,:,:]]*(len(val_labels)-1)).reshape((len(val_labels)-1),val_data.shape[1],val_data.shape[2],val_data.shape[3])),val_data[val_idx,:,:,:]]
        pred = model.predict(val_inputs)
        val_truth.append(val_labels[i])
        val_pred.append(pred_labels[np.argmax(pred)])
        if val_labels[i] == pred_labels[np.argmax(pred)]:
            n_corrects += 1
            
    accuracy = n_corrects/len(val_labels)*100
    
    return accuracy, val_truth, val_pred
    
    
def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename.encode(), fapl=propfaid)) as fid:
        return h5py.File(fid, mode)
    
    