import pandas as pd
import numpy as np
from scipy import signal
from numba import njit
from sklearn.model_selection import train_test_split
from numpy.random import randint
from itertools import compress

@njit
def calc_mag_diff(x):
    mag_diff = np.zeros(x.shape[0]-2)
    for i in range(1,x.shape[0]-1):
        diff = x[i,:]-x[i-1,:]
        mag_diff[i-1] = np.sqrt(np.power(diff[0],2)+np.power(diff[1],2)+np.power(diff[2],2))
    mag_diff = np.concatenate((mag_diff,np.zeros(2)))
    
    return mag_diff


def load_measurement(x,folder):
    samples = 100*5 # ten seconds interval
    subj = pd.read_csv(folder+x+".csv",usecols=(1,2,3))
    subj['mag_diff'] = calc_mag_diff(subj.values)
    subj['mag'] = subj.pow(2).sum(1).values
    subj['mag'] = np.sqrt(subj['mag'].values)
    psd = list()
    window = 'hann'
    for i in range(0,subj.values.shape[0],samples):
        fs, ps = signal.welch(subj['mag_diff'].values[i:i+samples-1], fs=50,window=window)
        ps = ps/np.max(ps)
        psd.append(ps)
    psd = np.concatenate((psd[:-1])).reshape(len(psd)-1,129) # 129 is (nperseg)/2 used for signal.welch function
    return psd


def load_subject(subject_id,ids_file,label_type,folder):
    ids_file = pd.read_csv(folder+ids_file)
    subject_measurements = list()
    measurements_labels = list()
    for measurement_id in ids_file[ids_file['subject_id'] == subject_id].values[:,0]:
        subject_measurements.append(load_measurement(measurement_id,folder))
        measurements_labels.append(ids_file[label_type][ids_file['measurement_id'] == measurement_id].values.astype(int))
    measurements_labels = np.stack(measurements_labels)
    return subject_measurements, measurements_labels


def threshold_data(data,labels,threshold=100):
    valid_data = list()
    valid_labels = list()
    for i in range(0,len(data)):
        if (data[i].shape[0] >= threshold):
            valid_data.append(data[i][:threshold,:])
            valid_labels.append(labels[i])
    return valid_data, valid_labels

def get_batch(data, labels):
    valid_data, valid_labels = threshold_data(data,labels)
    X_train, X_test, y_train, y_test = train_test_split(valid_data, valid_labels, test_size=0.25)
    X_train_m, X_train_u = get_pairs(X_train,y_train)
   
    return X_train_m, X_train_u, X_train, y_train, X_test, y_test

def get_pairs(data,labels):
    id_labels = [0,1,2,3,4]
    matched = dict()
    unmatched = dict()
    m_labels = dict()
    u_labels = dict()
    cat_data = dict()
    n = int(len(data)/4)
    for i in range(0,n):
        j = randint(0,len(data))
        matched[i] = data.pop(j)
        m_labels[i] = labels.pop(j)
        j = randint(0,len(data))
        unmatched[i] = data.pop(j)
        u_labels[i] = labels.pop(j)
    for i in range(0,5):
        cat_data[i] = list(compress(data,np.asarray(labels) == i))
    for i in range(0,n):
        j = m_labels[i][0]
        if len(cat_data[j])>0:
            m = cat_data[j].pop(0)
            matched[i] = np.stack((matched[i],m), axis=0)
        else:
            matched.pop(i)
    for i in range(0,n):
        c_labels = list(compress(id_labels,np.asarray(id_labels) != u_labels[i][0]))
        if len(cat_data[c_labels[0]])>0:
            u = cat_data[c_labels[0]].pop(0)
            unmatched[i] = np.stack((unmatched[i],u), axis=0)
        elif len(cat_data[c_labels[1]])>0:
            u = cat_data[c_labels[1]].pop(0)
            unmatched[i] = np.stack((unmatched[i],u), axis=0)
        elif len(cat_data[c_labels[2]])>0:
            u = cat_data[c_labels[2]].pop(0)
            unmatched[i] = np.stack((unmatched[i],u), axis=0)
        elif len(cat_data[c_labels[3]])>0:
            u = cat_data[c_labels[3]].pop(0)
            unmatched[i] = np.stack((unmatched[i],u), axis=0)
        else:
            unmatched.pop(i)
            
    return matched, unmatched


def get_pairs_alt(X,y):
    id_y = [0,1,2,3,4]
    l_matched = list()
    r_matched = list()
    l_unmatched = list()
    r_unmatched = list()
    m_y = list()
    u_y = list()
    cat_X = dict()
    n = int(len(X)/4)
    for i in range(0,n):
        j = randint(0,len(X))
        l_matched.append(X.pop(j))
        m_y.append(y.pop(j))
        j = randint(0,len(X))
        l_unmatched.append(X.pop(j))
        u_y.append(y.pop(j))
    for i in range(0,5):
        cat_X[i] = list(compress(X,np.asarray(y) == i))
    for i in range(0,n):
        j = m_y[i][0]
        if len(cat_X[j])>0:
            m = cat_X[j].pop(0)
            r_matched.append(m)
        else:
            l_matched.pop(i)
    for i in range(0,n):
        c_y = list(compress(id_y,np.asarray(id_y) != u_y[i][0]))
        if len(cat_X[c_y[0]])>0:
            u = cat_X[c_y[0]].pop(0)
            r_unmatched.append(u)
        elif len(cat_X[c_y[1]])>0:
            u = cat_X[c_y[1]].pop(0)
            r_unmatched.append(u)
        elif len(cat_X[c_y[2]])>0:
            u = cat_X[c_y[2]].pop(0)
            r_unmatched.append(u)
        elif len(cat_X[c_y[3]])>0:
            u = cat_X[c_y[3]].pop(0)
            r_unmatched.append(u)
        else:
            l_unmatched.pop(i)
    
    targets = np.hstack((np.ones(len(l_matched)),np.zeros(len(l_unmatched))))
    l_matched = np.stack(l_matched)
    l_unmatched = np.stack(l_unmatched)
    l_pairs = np.stack((l_matched,l_unmatched))
    r_matched = np.stack(r_matched)
    r_unmatched = np.stack(r_unmatched)
    r_pairs = np.stack((r_matched,r_unmatched))
    pairs = [l_pairs,r_pairs]
    
    return pairs, targets

#-----------------------------------------------------------------------------

subject_id = 1004
ids_file = "CIS-PD_Training_Data_IDs_Labels.csv"
folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"

data, labels = load_subject(subject_id,ids_file,'tremor',folder)

X_train_m, X_train_u, X_train, y_train, X_test, y_test = get_batch(data,labels)
