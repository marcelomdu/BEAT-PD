from utils import get_train_test, get_pairs, test_siamese
from model import get_siamese_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
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
    n_cross_val = 10
    ensembles = 10
    evaluate_every = 50 # interval for evaluating model
    #batch_size = 32
    n_iter = 1000 # No. of training iterations
    best = -1
    
    # Label to be trained (On_Off = 0, Dyskinesia = 1, Tremor = 2)
    tgt_label = 2
    n_samples = n_test[tgt_label]
    valid_subjects = [n_samples[k] != 0 for k in range(0,len(n_samples))]
    valid_subjects_ids = list(np.compress(valid_subjects,subjects_ids))
    
    weights_path = '/media/marcelomdu/Data/GIT_Repos/BEAT-PD/weights/'

    # model = get_siamese_model((100, 129, 1))
    # optimizer = Adam(lr = 0.00006)
    # model.compile(loss="binary_crossentropy",optimizer=optimizer)


    print("Start Training")
    print("-------------------------------------")
    
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
        for key in m_keys:
            d1 = f[str(subject_id)]['measurements'][str(key)][:,:][0,:,:]
            d1 = np.argmax(d1,axis=1)
            d1 = d1/np.max(d1)
            d1 = d1.reshape(d1.shape[0],1,1)
            data.append(d1)
            # d1 = f[str(subject_id)]['measurements'][str(key)][:,:][0,:,:]
            # d1 = d1.reshape(d1.shape[0],d1.shape[1],1)#np.moveaxis(d1,0,-1)
            # data.append(d1)
            #data.append(np.moveaxis(f[str(subject_id)]['measurements'][str(key)][:,:],0,-1))
        labels = f[str(subject_id)]['labels'][:][:,tgt_label]
        t_start = time.time()

        # Initialize model
        model = get_siamese_model((150, 1, 1))
        if best>0:
            model.set_weights(weights)
        optimizer = Adam(lr = 0.00006)
        model.compile(loss="binary_crossentropy",optimizer=optimizer)


        for k in range(0,n_cross_val):

            # Get train and test data
            X_train, X_test, y_train, y_test = get_train_test(data, labels, 
                                                            classes=classes, 
                                                            n_tests=n_tests, 
                                                            categorical=False, 
                                                            th_value=150, 
                                                            balance=True)
                
            # Train
            for j in range(0, ensembles):
                inputs, targets = get_pairs(X_train,y_train)
                for i in range(1, n_iter+1):
                    # print("Cross {0} - Ensemble {1} - Iter {2}".format(k,j,i))
                    # print("\n ------------- ")
                    loss = model.train_on_batch(inputs, targets)
                    # print("Train Loss: {0}".format(loss)+" Iter: {}".format(i))
                    if i % evaluate_every == 0:
                        print("\n ------------- ")
                        print("Cross {0} - Ensemble {1} - Iter {2}".format(k,j,i))
                        print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
                        val_acc, val_truth, val_pred = test_siamese(model,val_data=X_test,val_labels=y_test)
                        print('tru: {0}'.format(val_truth))
                        print('pre: {0}'.format(val_pred))
                        print('accuracy: {0}'.format(val_acc))
                        if val_acc >= best:
                            print("Current best: {0}, previous best: {1}".format(val_acc, best))
                            model.save_weights(os.path.join(weights_path, 'weights.{}.h5'.format(subject_id)))
                            model.save_weights(os.path.join(weights_path, 'weights.best.h5'))
                            weights = model.get_weights()
                            best = val_acc
            
        
        


