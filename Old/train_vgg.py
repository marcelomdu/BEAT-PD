import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from model import CNN_2D, VGG16
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from metrics import *
import os


#Force CPU use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def load_data(id):
    #Load file and get keys (subject ids)
    folder = '/home/marcon/Documents/Challenges/DREAM/'
    # folder = '/usr/share/datasets/BEAT-PD/'
    filename = 'training_data_PSD.hdf5'
    filepath = folder+filename
    file = h5py.File(filepath,'r') 
    fkeys = list(file.keys()) 

    #Set target subject
    id_index = fkeys.index(id)
    example_ids = list(file[fkeys[id_index]]['measurements'].keys())
    spect_ids = list(file[fkeys[id_index]]['measurements'].keys())[int(len(example_ids)/2):]  

    #Load file data
    fdata = []
    for example in spect_ids:
        fdata.append(file[id]['measurements'][example][:,:])

    #Make examples the same shape
    data = []
    labels = []
    threshold_len = 200
    for n in range(len(fdata)):
        #Put channels in third dimension
        fdata[n] = np.swapaxes(fdata[n],0,2)
        #Keep examples with at least threshold length, limiting their length to threshold
        if len(fdata[n][0]) > threshold_len:
            fdata[n] = fdata[n][:,:200,:]
            data.append(fdata[n])
            labels.append(n)

    #Load labels
    flabels = []
    for n in labels:
        flabels.append(file[id]['labels'][n][:])

    #Get labels for on_off classification
    on_off = []
    for item in flabels:
        on_off.append(item[0])

    #Remove nan values:
    nan_index = [n for n in range(len(on_off)) if on_off[n] < 0]
    nan_index = sorted(nan_index, reverse=True)
    for item in nan_index:
        on_off.pop(item)
        data.pop(item)

    #One-hot encoding
    on_off = to_categorical(on_off, num_classes=5)

    return data, on_off

def train_model(X_train, X_test, y_train, y_test):

    batch_size = 4
    epochs = 80
    learning_rate = 1e-5

    optimizer = Adam(lr = learning_rate)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    
    # Initialize model
    print("\nTraining on subject: {}".format(id))    
    model = VGG16(input_shape=(91,200,3), num_classes=5)
    model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=['accuracy',precision_m,recall_m,f1_m])
    model.fit(X_train,y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[es])
    
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test precision:', score[2])
    print('Test recall:', score[3])
    print('Test f1:', score[4])


if __name__ == "__main__":
    id = '1006'
    data, labels = load_data(id) 
    
    X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.2, \
                                                            random_state=42) 
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    train_model(X_train, X_test, y_train, y_test)
