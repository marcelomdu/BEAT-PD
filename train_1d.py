from utils import get_train_test, get_pairs
from model import get_zhang_model
from tensorflow.keras.optimizers import Adam
from utils import hdf5_handler
import numpy as np

import os
import time
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lsa', action='store_true', default=False,
            help='uses lsa path for hdf5')

    args = parser.parse_args()

    subjects_ids = [1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051]
    n_test = [[4,2,0,2,2,4,14,2,6,5,2,10,0,4,5,0],
              [4,0,4,2,0,7,0,2,9,8,3,2,0,12,4,0],
              [2,2,2,2,3,8,22,3,23,0,2,0,4,6,7,0]]
    n_labels = [[[0,1,2,3],[0,1],[],[0,2,3],[0,1,2,4],[1,2,3,4],[0,1,2],[0,1,2,3],[0,2,4],[0,1,2],[0,1,2,3],[0,1],[],[1,2,3],[1,2,3],[]],
                [[0,1,2,3],[],[0,1],[0,1,2],[],[0,1,2],[],[0,1,2],[0,2],[0,1,2],[0,1],[0,1],[],[2,3],[0,1,2],[]],
                [[0,1,2,3],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1],[0,1],[0,1],[1,2],[],[0,1,2,3],[],[0,1],[2,3],[1,2,3],[]]]
    
    ids_file = "CIS-PD_Training_Data_IDs_Labels.csv"
    folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
    if args.lsa:
        folder = "/home/marcelomd/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
    f = hdf5_handler(folder+'training_data_PSD.hdf5','a')

    # Hyper parameters
    n_cross_val = 1
    ensembles = 10
    evaluate_every = 50 # interval for evaluating model
    batch_size = 100
    epochs = 1000
    n_iter = 10000 # No. of training iterations
    best = -1
    
    # Label to be trained (On_Off = 0, Dyskinesia = 1, Tremor = 2)
    tgt_label = 2
    n_samples = n_test[tgt_label]
    valid_subjects = [n_samples[k] != 0 for k in range(0,len(n_samples))]
    valid_subjects_ids = list(np.compress(valid_subjects,subjects_ids))
    
    # Binary classifier test
    valid_subjects_ids = [1004]

    weights_path = '/media/marcelomdu/Data/GIT_Repos/BEAT-PD/weights/'

    # model = get_siamese_model((100, 129, 1))
    # optimizer = Adam(lr = 0.00006)
    # model.compile(loss="binary_crossentropy",optimizer=optimizer)

    print("Start Training")
    print("-------------------------------------")
    
    for subject_id in subjects_ids:
        
        # Get valid classes for subject
        idx = [subjects_ids[i] == subject_id for i in range(0,len(subjects_ids))]
        classes = list(np.compress(idx, n_labels[tgt_label]))[0]
        num_classes = len(classes)
        n_tests = np.stack(n_samples)[idx][0]

        # Load data
        m_keys = list((f[str(subject_id)]['measurements']).keys())
        m_keys = m_keys[:int(len(m_keys)/2)]
        data = list()
        for key in m_keys:
            d1 = f[str(subject_id)]['measurements'][str(key)][:,:][0,:,:]
            d1 = np.argmax(d1,axis=1)
            d1 = d1/np.max(d1)
            d1 = d1.reshape(d1.shape[0],1,1)
            data.append(d1)
        labels = f[str(subject_id)]['labels'][:][:,tgt_label]
        t_start = time.time()
        
        # Initialize model
        model = get_zhang_model((200,1,1),num_classes)
        optimizer = Adam(lr = 0.00006)
        model.compile(loss="categorical_crossentropy",
                    optimizer=optimizer, 
                    metrics=['accuracy'])

        t_start = time.time()
        X_train, X_test, y_train, y_test = get_train_test(data, labels, 
                                                            classes=classes, 
                                                            n_tests=n_tests, 
                                                            num_classes=num_classes,
                                                            categorical=True, 
                                                            th_value=200, 
                                                            balance=True)

        # acc_file = open(os.path.join(weights_path,'{}_acc.txt'.format(subject_id),'a'))

        model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test,y_test),
                shuffle=True)

        scores = model.evaluate(X_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        
        # acc_file.close()
        
        


