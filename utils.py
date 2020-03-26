import numpy as np
import h5py
import contextlib
import copy
from sklearn.model_selection import train_test_split
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
        if (data[i].shape[1] >= threshold):
            valid_data.append(data[i][:,-threshold:,:])
            valid_labels.append(labels[i])
    return valid_data, valid_labels

def get_train_test(data, labels, dim='3D', categorical=True, classes=[], num_samples=0, num_classes=5, threshold=True, th_value=100, balance=False):
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
        X_train, X_test, y_train, y_test, _ = get_balanced_data(valid_data,valid_labels,classes,num_samples)
    else:
        X_train, X_test, y_train, y_test = train_test_split(valid_data, valid_labels, test_size=0.25)
        X_train = np.stack(X_train)
    
    X_test = np.stack(X_test)

    if categorical:
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
                k1 = randint(0,len(X[l]))
                l_unmatched.append(X[l].pop(k1))
                k2 = randint(0,len(l_choose))
                l2 = l_choose[int(k2)]
                if len(X[l2])>0:
                    k1 = randint(0,len(X[l2]))
                    r_unmatched.append(X[l2].pop(k1))
                    x_len[int(k2)] -= 1
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

def get_balanced_data(data,labels,classes,num_samples):
    
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
        for _ in range(0, num_samples):
            k = randint(0,len(cat_data[i]))
            X = cat_data[i].pop(k)
            X_test.append(X)
            y_test.append(i)
        for _ in range(0, min(len(cat_data[i]),5*num_samples)):
            k = randint(0,len(cat_data[i]))
            X = cat_data[i].pop(k)
            X_train[i].append(X)
            y_train[i].append(i)


    return X_train, X_test, y_train, y_test, cat_data       


def test_siamese(model, train_data, val_data, train_labels, val_labels):
    
    n_labels = list()
    train_data_list = list()
    
    for i in range(0,5):
        train_data_list.append(list(compress(train_data,np.asarray(train_labels)==i)))
        n_labels.append(len(train_data_list[i]))
        if n_labels[i]>0:
            train_data_list[i] = np.stack(train_data_list[i])
            train_data_list[i] = train_data_list[i].reshape(train_data_list[i].shape[0],train_data_list[i].shape[1],train_data_list[i].shape[2],train_data_list[i].shape[3])

    val_data = np.stack(val_data)

    n_correct = 0

    for i in range(0,val_data.shape[0]):
        preds = list()
        for j in range(0,len(n_labels)):
            if n_labels[j]>0:
                inputs = [(np.asarray([val_data[i,:,:,:]]*n_labels[j]).reshape(n_labels[j],val_data.shape[1],val_data.shape[2],val_data.shape[3])),train_data_list[j]]
                pred = np.sum(model.predict(inputs))/n_labels[j]
            else:
                pred = -1
            preds.append(pred)
        if np.argmax(np.stack(preds))==val_labels[i]:
            n_correct += 1

    accuracy = n_correct/len(val_labels)

    return accuracy
    
    
def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename.encode(), fapl=propfaid)) as fid:
        return h5py.File(fid, mode)
    
    
#------------------------------------------------------------------------------------
# Old func defs

#def make_oneshot_task(val_data,val_labels,N):
#    """Create pairs of test image, support set for testing N way one-shot learning. """
#    _, w, h = val_data.shape
#    
#    test_label = randint(0,4)
#    
#    # Matched and unmatched candidates
#    i_m = (np.asarray(val_labels)==test_label)[:,0]
#    i_u = (np.asarray(val_labels)!=test_label)[:,0]
#    m_candidates = val_data[i_m,:,:]
#    u_candidates = val_data[i_u,:,:]
#    
#    # Random indices for sampling
#    try:
#        m_idx1, m_idx2 = choice(m_candidates.shape[0],replace=False,size=(2,)) # Non repetitive random indices
#        u_indices = randint(0,u_candidates.shape[0],size=(N,))
#    except:
#        print('Not enough validation candidates')
#
#    # Matched image from support_set will be allocated to position '0' then shuffled
#    test_image = np.asarray([m_candidates[m_idx1,:,:]]*N).reshape(N, w, h, 1)
#    support_set = u_candidates[u_indices,:,:]
#    support_set[0,:,:] = m_candidates[m_idx2,:,:]
#    support_set = support_set.reshape(N, w, h, 1)
#    targets = np.zeros((N,))
#    targets[0] = 1
#    targets, test_image, support_set = shuffle(targets, test_image, support_set)
#
#    pairs = [test_image, support_set]
#
#    return pairs, targets
#
#  
#def test_oneshot(model,val_data,val_labels,N,k, verbose = 0):
#    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
#    n_correct = 0
#    if verbose:
#        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
#    for _ in range(k):
#        # try:
#        inputs, targets = make_oneshot_task(val_data,val_labels,N)
#        probs = model.predict(inputs)
#        if np.argmax(probs) == np.argmax(targets):
#            n_correct+=1
#        # except:
#        #     print('Not enough validation candidates')
#    percent_correct = (100.0 * n_correct / k)
#    if verbose:
#        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
#    return percent_correct
