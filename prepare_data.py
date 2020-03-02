import pandas as pd
import numpy as np
import pickle
import os
from scipy import signal
from numba import njit
from numpy.random import randint


@njit
def calc_mag_diff(x):
    mag_diff = np.zeros(x.shape[0]-2)
    for i in range(1,x.shape[0]-1):
        diff = x[i,:]-x[i-1,:]
        mag_diff[i-1] = np.sqrt(np.power(diff[0],2)+np.power(diff[1],2)+np.power(diff[2],2))
    mag_diff = np.concatenate((mag_diff,np.zeros(2)))
    
    return mag_diff


def load_measurement(x,folder,interval=10):
    samples = interval*50 # ten seconds interval from a 50 Hz sample rate
    subj = pd.read_csv(folder+x+".csv",usecols=(1,2,3))
    subj['mag_diff'] = calc_mag_diff(subj.values)
    subj['mag'] = subj.pow(2).sum(1).values
    subj['mag'] = np.sqrt(subj['mag'].values)
    psd = list()
    window = 'hann'
    for i in range(0,subj.values.shape[0],samples):
        if subj['mag_diff'].values[i:i+samples-1].shape[0]>255: # 255 is the default nperseg from signal.welch
            _, ps = signal.welch(subj['mag_diff'].values[i:i+samples-1], fs=50,window=window)
            #ps = ps[29:]
            ps = ps/np.max(ps)
            psd.append(ps)
    psd = np.concatenate((psd[:-1])).reshape(len(psd)-1,129) # 129 is ((nperseg)/2)+1 used for signal.welch function
    
    return psd


def load_subject(subject_id,ids_file,folder):
    ids_file = pd.read_csv(folder+ids_file)
    subject_measurements = list()
    measurements_labels_medication = list()
    measurements_labels_dyskinesia = list()
    measurements_labels_tremor = list()
    measurements_labels = dict()
    for measurement_id in ids_file[ids_file['subject_id'] == subject_id].values[:,0]:
        subject_measurements.append(load_measurement(measurement_id,folder))
        measurements_labels_medication.append(ids_file['on_off'][ids_file['measurement_id'] == measurement_id].values.astype(int))
        measurements_labels_dyskinesia.append(ids_file['dyskinesia'][ids_file['measurement_id'] == measurement_id].values.astype(int))
        measurements_labels_tremor.append(ids_file['tremor'][ids_file['measurement_id'] == measurement_id].values.astype(int))
    measurements_labels['medication'] = np.stack(measurements_labels_medication)
    measurements_labels['dyskinesia'] = np.stack(measurements_labels_dyskinesia)
    measurements_labels['tremor'] = np.stack(measurements_labels_tremor)
    
    return subject_measurements, measurements_labels


def load_subjects(subjects_ids,ids_file,folder):
    data = dict()
    for subject_id in subjects_ids:
        data[subject_id] = load_subject(subject_id,ids_file,folder)
    
    return data


#-----------------------------------------------------------------------------
    
if __name__ == '__main__':

    subjects_ids = [1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051]
    ids_file = "CIS-PD_Training_Data_IDs_Labels.csv"
    folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"

    data = dict()
    for subject_id in subjects_ids:
        print('Loading subject '+str(subject_id))
        data[subject_id] = load_subject(subject_id,ids_file,folder)

    with open(os.path.join(folder,"training_data.pickle"), "wb") as f:
        pickle.dump(data,f)
    
    print('Prepare data done!')