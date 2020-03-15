import pandas as pd
import numpy as np
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


def load_measurement(x,folder):
    samples = 100*5 # ten seconds interval
    subj = pd.read_csv(folder+x+".csv",usecols=(1,2,3))
    subj['mag'] = subj.pow(2).sum(1).values
    subj['mag'] = np.sqrt(subj['mag'].values)
    subj['mag_diff'] = calc_mag_diff(subj.values)
    psd = list()
    window = 'hann'
    for i in range(0,subj.values.shape[0],samples):
        _, ps = signal.welch(subj['mag_diff'].values[i:i+samples-1], fs=50,window=window)
        #ps = ps[29:]
        ps = ps/np.max(ps)
        psd.append(ps)
    psd = np.concatenate((psd[:-1])).reshape(len(psd)-1,129) # 129 is (nperseg)/2 used for signal.welch function
    
    return psd


def load_subject(subject_id,ids_file,label_type,folder):
    ids_file = pd.read_csv(folder+ids_file)
    subject_measurements = list()
    measurements_labels = list()
    for measurement_id in ids_file[ids_file['subject_id'] == subject_id].values[:,0]:
        subject_measurements.append(load_measurement(measurement_id,folder))
        measurements_labels.append(ids_file[label_type][ids_file['measurement_id'] == measurement_id].values.astype(int))
    measurements_labels = np.stack(measurements_labels)
    
    return subject_measurements, measurements_labels
