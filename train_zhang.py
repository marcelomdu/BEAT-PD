from utils import get_train_test, get_pairs
from model import get_zhang_model
from tensorflow.keras.optimizers import Adam
from utils import hdf5_handler

import os
import time


if __name__ == '__main__':

    subjects_ids = [1004]#[1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051]
    ids_file = "CIS-PD_Training_Data_IDs_Labels.csv"
    folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
    f = hdf5_handler(folder+'training_data_PSD.hdf5','a')
    num_classes = 5

    # Hyper parameters
    batch_size = 10
    epochs = 10000 # No. of training iterations
    

    weights_path = '/media/marcelomdu/Data/GIT_Repos/BEAT-PD/weights/'

    print("Start Training")
    print("-------------------------------------")
    
    for subject_id in subjects_ids:
        # Initialize model
        model = get_zhang_model((3,200,91),num_classes)
        optimizer = Adam(lr = 0.0001)
        model.compile(loss="categorical_crossentropy",
                    optimizer=optimizer, 
                    metrics=['accuracy'])

        # Load data
        m_keys = list((f[str(subject_id)]['measurements']).keys())
        m_keys = m_keys[:int(len(m_keys)/2)]
        data = list()
        for key in m_keys:
            data.append(f[str(subject_id)]['measurements'][str(key)][:,:])
        labels = list(f[str(subject_id)]['labels'][:][:,2])
        t_start = time.time()
        X_train, X_test, y_train, y_test = get_train_test(data, labels, 
                                                        dim='3D',
                                                        categorical=True,
                                                        num_classes=num_classes,
                                                        th_value=200)

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
        
        


