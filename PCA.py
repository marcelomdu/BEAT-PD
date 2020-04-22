from utils import get_train_test, get_pairs
from model import get_zhang_model
from tensorflow.keras.optimizers import Adam
from utils import hdf5_handler, get_train_test
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
        data = np.stack(data)
        labels = np.stack(labels)
        
    y = pd.DataFrame(data=labels,columns=['target'])
    x = StandardScaler().fit_transform(data)
    pca = PCA(n_components=10)
    principalComponents = pca.fit_transform(x)
    
    principalDf = pd.DataFrame(data = principalComponents[:,1:3], 
                               columns = ['principal component 1', 'principal component 2'])
    
    finalDf = pd.concat([principalDf, y], axis = 1)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [0,1,2,3]
    colors = ['r', 'g', 'b', 'y']
    
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
