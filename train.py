from load_data import load_subject
from utils import get_batch, test_oneshot
from model import get_siamese_model
from tensorflow.keras.optimizers import Adam

import os
import time


subject_id = 1007
ids_file = "CIS-PD_Training_Data_IDs_Labels.csv"
folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"

data, labels = load_subject(subject_id,ids_file,'tremor',folder)

#pairs, targets, X_train, y_train, X_test, y_test = get_batch(data, labels)

# Hyper parameters
evaluate_every = 2 # interval for evaluating on one-shot tasks
#batch_size = 32
n_iter = 10 # No. of training iterations
N_way = 3 # how many classes for testing one-shot tasks
n_val = 250 # how many one-shot tasks to validate on
best = -1


model_path = '/media/marcelomdu/Data/GIT_Repos/BEAT-PD/BEAT-PD/weights/'

model = get_siamese_model((100, 129, 1))
optimizer = Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer)


print("Starting training process!")
print("-------------------------------------")
t_start = time.time()
for i in range(1, n_iter+1):
    inputs, targets, X_train, y_train, X_test, y_test = get_batch(data, labels)
    loss = model.train_on_batch(inputs, targets)
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
        print("Train Loss: {0}".format(loss)) 
        val_acc = test_oneshot(model, X_test, y_test, N_way, n_val, verbose=True)
        model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            best = val_acc
        

