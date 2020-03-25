from utils import get_train_test, get_pairs, test_siamese
from model import get_siamese_model
from tensorflow.keras.optimizers import Adam
from utils import hdf5_handler
import numpy as np

import os
import time


if __name__ == '__main__':

    subjects_ids = [1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051]
    n_test = [[4,2,4,2,2,4,14,2,6,5,2,10,0,4,5,4],
              [4,0,4,2,0,7,0,2,9,8,3,2,0,12,4,0],
              [2,2,2,2,3,8,22,3,23,0,2,0,4,6,7,0]]
    ids_file = "CIS-PD_Training_Data_IDs_Labels.csv"
    folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
    f = hdf5_handler(folder+'training_data_PSD.hdf5','a')

    # Hyper parameters
    evaluate_every = 1000 # interval for evaluating model
    #batch_size = 32
    n_iter = 10000 # No. of training iterations
    best = -1
    
    # Label to be trained (On_Off = 0, Dyskinesia = 1, Tremor = 2)
    l_test = 2


    weights_path = '/media/marcelomdu/Data/GIT_Repos/BEAT-PD/weights/'

    # model = get_siamese_model((100, 129, 1))
    # optimizer = Adam(lr = 0.00006)
    # model.compile(loss="binary_crossentropy",optimizer=optimizer)


    print("Start Training")
    print("-------------------------------------")
    
    for subject_id in subjects_ids:
        # Initialize model
        model = get_siamese_model((3,200, 91))
        if best>0:
            model.set_weights(weights)
        optimizer = Adam(lr = 0.00006)
        model.compile(loss="binary_crossentropy",optimizer=optimizer)

        valid_samples = [n_test[l_test][k] != 0 for k in range(0,len(n_test[l_test]))]

        # Load data
        m_keys = list((f[str(subject_id)]['measurements']).keys())
        m_keys = m_keys[:int(len(m_keys)/2)]
        m_keys = list(np.compress(valid_samples,m_keys))
        data = list()
        for key in m_keys:
            data.append(f[str(subject_id)]['measurements'][str(key)][:,:])
        labels = f[str(subject_id)]['labels'][:][:,l_test]
        labels = labels[valid_samples]
        t_start = time.time()
        X_train, X_test, y_train, y_test = get_train_test(data, labels, categorical=False)

        acc_file = open(os.path.join(weights_path,'{}_acc.txt'.format(subject_id),'a'))

        # Train
        for i in range(1, n_iter+1):
            inputs, targets = get_pairs(X_train,y_train)
            loss = model.train_on_batch(inputs, targets)
            print("Train Loss: {0}".format(loss)+" Iter: {}".format(i))
            if i % evaluate_every == 0:
                print("\n ------------- ")
                print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
                val_acc = test_siamese(model, X_train, X_test, y_train, y_test)
                acc_file.write(str(val_acc)+',')
                print('accuracy: {0}'.format(val_acc))
                if val_acc >= best:
                    print("Current best: {0}, previous best: {1}".format(val_acc, best))
                    model.save_weights(os.path.join(weights_path, 'weights.{}.h5'.format(subject_id)))
                    model.save_weights(os.path.join(weights_path, 'weights.best.h5'))
                    weights = model.get_weights()
                    best = val_acc
        
        acc_file.close()
        
        


