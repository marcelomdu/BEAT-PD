from load_data import load_subject
from utils import get_batch, test_model
from model import get_siamese_model
from tensorflow.keras.optimizers import Adam
from utils import hdf5_handler

import os
import time
import pickle


if __name__ == '__main__':

    subjects_ids = [1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051]
    ids_file = "CIS-PD_Training_Data_IDs_Labels.csv"
    folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
    f = hdf5_handler(folder+'training_data.hdf5','a')

    #pairs, targets, X_train, y_train, X_test, y_test = get_batch(data, labels)

    # Hyper parameters
    evaluate_every = 20 # interval for evaluating on one-shot tasks
    #batch_size = 32
    n_iter = 2000 # No. of training iterations
    #N_way = 3 # how many classes for testing one-shot tasks
    #n_val = 250 # how many one-shot tasks to validate on
    best = -1


    model_path = '/media/marcelomdu/Data/GIT_Repos/BEAT-PD/weights/'

    model = get_siamese_model((100, 129, 1))
    optimizer = Adam(lr = 0.001) #0.00006
    model.compile(loss="binary_crossentropy",optimizer=optimizer)


    print("Start Training")
    print("-------------------------------------")
    
    for subject_id in subjects_ids:
        data = f[str(subjects_ids)]['measurements'][:][:,:]
        labels = f[str(subjects_ids)]['labels'][:]
        t_start = time.time()
        for i in range(1, n_iter+1):
            inputs, targets, X_train, y_train, X_test, y_test = get_batch(data, labels)
            loss = model.train_on_batch(inputs, targets)
            print("Train Loss: {0}".format(loss)+" Iter: {}".format(i))
            if i % evaluate_every == 0:
                print("\n ------------- \n")
                print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
                print("Train Loss: {0}".format(loss)) 
                val_acc = test_model(model, X_test, y_test)
                model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
                if val_acc >= best:
                    print("Current best: {0}, previous best: {1}".format(val_acc, best))
                    best = val_acc


