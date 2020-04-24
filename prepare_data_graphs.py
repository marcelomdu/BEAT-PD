import os
import argparse
import pandas as pd
import numpy as np
from scipy import signal
from numba import jit, njit
from utils_graph import hdf5_handler
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--ids_file', type=str, default=None,
                    help='.csv file with measurements IDs')
parser.add_argument('--folder', type=str, default=None,
                    help='Folder with the measurements files and ids_file')
parser.add_argument('--prefix', type=str, default=None, 
                    help='HDF5 destination file name')

@njit
def calc_mag_diff(x):
    mag_diff = np.zeros(x.shape[0]-2)
    for i in range(1,x.shape[0]-1):
        diff = x[i,:]-x[i-1,:]
        mag_diff[i-1] = np.sqrt(np.power(diff[0],2)+np.power(diff[1],2)+np.power(diff[2],2))
    mag_diff = np.concatenate((mag_diff,np.zeros(2)))
    
    return mag_diff

@jit
def calc_cn_matrix(x,p_value):
    n = x.shape[0]
    cn_matrix = np.zeros((n,n))
    p_matrix = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            cn_matrix[i,j], p_matrix[i,j] = pearsonr(x[i,:], x[j,:])
    cn_matrix[p_matrix > p_value] = 0
    
    return cn_matrix


def load_spectrums(x,folder,interval=5):
    sos = signal.butter(10,4,btype='high',fs=50,output='sos')
    samples = interval*50 # interval in seconds
    subj = pd.read_csv(folder+x+".csv")
    if 'x' in subj.columns:
        subj = subj[['x','y','z']]
    elif 'X' in subj.columns:
        subj = subj[['X','Y','Z']]
    subj['mag_diff'] = calc_mag_diff(subj.values[:,0:3])
    subj['mag_diff_f'] = signal.sosfilt(sos,subj['mag_diff'].values)
    nperseg = samples
    tau = nperseg/5
    window = signal.windows.exponential(nperseg,tau=tau)

    # Calculates the Power Spectrum Density by Welch's method and its 1st and 2nd deltas
    if subj['mag_diff_f'].values.shape[0] > samples:
        _, psd = signal.welch(subj['mag_diff_f'].values,fs=50,window=window,detrend='linear')
        max_psd = np.max(psd)
        psd = psd/max_psd
        d1psd = np.gradient(psd)
        d2psd = np.gradient(d1psd)
    else:
        psd=d1psd=d2psd=max_psd = None
        
    return psd, d1psd, d2psd, max_psd


def load_subject(subject_id,ids_file,folder):
    id_file = pd.read_csv(folder+ids_file)
    subject_psds = list()
    subject_d1psds = list()
    subject_d2psds = list()
    subject_max_psds = list()
    labels_med = list()
    labels_dys = list()
    labels_tre = list()
    valid_measurements = os.listdir(os.fsencode(folder))
    for measurement_id in id_file[id_file['subject_id'] == subject_id].values[:,0]:
        if os.fsencode(measurement_id+'.csv') in valid_measurements:
            subject_psd, subject_d1psd, subject_d2psd, subject_max_psd = load_spectrums(measurement_id,folder)
            if subject_psd is not None:
                subject_psds.append(subject_psd)
                subject_d1psds.append(subject_d1psd)
                subject_d2psds.append(subject_d2psd)
                subject_max_psds.append(subject_max_psd)
                labels_med.append(id_file['on_off'][id_file['measurement_id'] == measurement_id].values.astype(int)[0])
                labels_dys.append(id_file['dyskinesia'][id_file['measurement_id'] == measurement_id].values.astype(int)[0])
                labels_tre.append(id_file['tremor'][id_file['measurement_id'] == measurement_id].values.astype(int)[0])
    subject_psds = np.stack(subject_psds)
    subject_d1psds = np.stack(subject_d1psds)
    subject_d2psds = np.stack(subject_d2psds)
    
    p_value = 0.001
    
    cn_matrix1 = calc_cn_matrix(subject_psds,p_value)
    cn_matrix2 = calc_cn_matrix(subject_d1psds,p_value)
    cn_matrix3 = calc_cn_matrix(subject_d2psds,p_value)
    
    labels_med = np.stack(labels_med).reshape(-1,1)
    labels_dys = np.stack(labels_dys).reshape(-1,1)
    labels_tre = np.stack(labels_tre).reshape(-1,1)
    
    labels = np.hstack((labels_med,labels_dys))
    labels = np.hstack((labels,labels_tre))

    enc=OneHotEncoder(handle_unknown='ignore',sparse=False)
    ft_matrix1 = enc.fit_transform(labels_med)
    ft_matrix2 = enc.fit_transform(labels_dys)
    ft_matrix3 = enc.fit_transform(labels_tre)
    
    ft_matrix4 = np.stack(subject_max_psds)
    ft_matrix4 = ft_matrix4/np.max(ft_matrix4)
    
    return cn_matrix1, cn_matrix2, cn_matrix3, ft_matrix1, ft_matrix2, ft_matrix3, ft_matrix4, labels

#-----------------------------------------------------------------------------

args = parser.parse_args()

if __name__ == '__main__':

    if (args.ids_file is not None) and (args.folder is not None) and (args.prefix is not None):

        ids_file = args.ids_file
        folder = args.folder
        hdf5_prefix = args.prefix
        subjects_ids = np.unique(pd.read_csv(folder+ids_file,usecols=[1]).values).tolist()
        
        f = hdf5_handler(folder+hdf5_prefix+'.hdf5','a')
    
        for subject_id in subjects_ids:
            print('Loading subject '+str(subject_id))
            data = load_subject(subject_id,ids_file,folder)
            subj = f.create_group(str(subject_id))
            subj.create_dataset('cn_matrix1',data=data[0])
            subj.create_dataset('cn_matrix2', data=data[1])
            subj.create_dataset('cn_matrix3', data=data[2])
            subj.create_dataset('ft_matrix1', data=data[3])
            subj.create_dataset('ft_matrix2', data=data[4])
            subj.create_dataset('ft_matrix3', data=data[5])
            subj.create_dataset('ft_matrix4', data=data[6])
            subj.create_dataset('labels', data=data[7])
            
        print('Prepare data done!')
    
    else:
        print('Args missing')
    


