from utils import get_train_test, get_pairs
from model import get_zhang_model
from tensorflow.keras.optimizers import Adam
from utils import hdf5_handler, get_train_test, dict_to_array
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib import pyplot as plt

import os
import time
import argparse


if __name__ == '__main__':

    subjects_ids = [1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051]
    n_test = [[4,2,0,2,2,4,14,2,6,5,2,10,0,4,5,0],
              [4,0,4,2,0,7,0,2,9,8,3,2,0,12,4,0],
              [2,2,2,2,3,8,22,3,23,0,2,0,4,6,7,0]]
    n_labels = [[[0,1,2,3],[0,1],[],[0,2,3],[0,1,2,4],[1,2,3,4],[0,1,2],[0,1,2,3],[0,2,4],[0,1,2],[0,1,2,3],[0,1],[],[1,2,3],[1,2,3],[]],
                [[0,1,2,3],[],[0,1],[0,1,2],[],[0,1,2],[],[0,1,2],[0,2],[0,1,2],[0,1],[0,1],[],[2,3],[0,1,2],[]],
                [[0,1,2,3],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1],[0,1],[0,1],[1,2],[],[0,1,2,3],[],[0,1],[2,3],[1,2,3],[]]]
    
    ids_file = "CIS-PD_Training_Data_IDs_Labels.csv"
    folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
    f = hdf5_handler(folder+'training_data_PSD.hdf5','a')

    # Label to be trained (On_Off = 0, Dyskinesia = 1, Tremor = 2)
    tgt_label = 2
    n_samples = n_test[tgt_label]
    valid_subjects = [n_samples[k] != 0 for k in range(0,len(n_samples))]
    valid_subjects_ids = list(np.compress(valid_subjects,subjects_ids))
    
    # Select one subject
    valid_subjects_ids = [1004]

    for subject_id in valid_subjects_ids:
        
        # Get valid classes for subject
        idx = [subjects_ids[i] == subject_id for i in range(0,len(subjects_ids))]
        classes = list(np.compress(idx, n_labels[tgt_label]))[0]
        num_classes = len(classes)
        n_tests = np.stack(n_samples)[idx][0]

        # Load data
        m_keys = list((f[str(subject_id)]['measurements']).keys())
        m_keys = m_keys[:int(len(m_keys)/2)]
        data = list()
        labels = list()
        i = 0
        for key in m_keys:
            d1 = f[str(subject_id)]['measurements'][str(key)][:,:][0,:,:]
            d1 = np.argmax(d1,axis=1)
            d1 = d1/np.max(d1)
            if len(d1)>200:
                data.append(d1[-200:])
                l = f[str(subject_id)]['labels'][:][:,tgt_label][i]
                labels.append(l)
            i += 1
        # x = np.stack(data)
        # x = StandardScaler().fit_transform(x)
        # y = np.stack(labels)
        
        X_train, X_test, y_train, y_test = get_train_test(data, labels, 
                                                    classes=classes, 
                                                    n_tests=n_tests, 
                                                    num_classes=num_classes,
                                                    categorical=False,
                                                    balance=True,
                                                    threshold=False)
        X_train, y_train = dict_to_array(X_train, y_train)

        clf = SVC(degree=10)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
