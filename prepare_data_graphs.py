import os
import argparse
import pandas as pd
import numpy as np
from scipy import signal
from numba import jit, njit
from utils_graph import hdf5_handler
from scipy.stats import pearsonr, describe, iqr, entropy, tsem, median_absolute_deviation
from sklearn.preprocessing import OneHotEncoder, StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--study', type=str, default="CIS",
                    help='Study name')
parser.add_argument('--prefix', type=str, default="train_test_data_graphs", 
                    help='HDF5 destination file name')
parser.add_argument('--ignore_test_data', default=False, action='store_true',
                    help='Include test data')


@njit
def calc_mag_diff(x):
    mag_diff = np.zeros(x.shape[0]-2)
    for i in range(1,x.shape[0]-1):
        diff = x[i,:]-x[i-1,:]
        mag_diff[i-1] = np.sqrt(np.power(diff[0],2)+np.power(diff[1],2)+np.power(diff[2],2))
    mag_diff = np.concatenate((mag_diff,np.zeros(2)))
    
    return mag_diff

@jit(nopython=False)
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
    
    t_describe = describe(subj['mag_diff_f'])
    t_min = t_describe[1][0]
    t_max = t_describe[1][1]
    t_mean = t_describe[2]
    t_var = t_describe[3]
    t_skew = t_describe[4]
    t_kurtosis = t_describe[5]
    t_iqr = iqr(subj['mag_diff_f'])
    t_sem = tsem(subj['mag_diff_f'])
    t_mad = median_absolute_deviation(subj['mag_diff_f'])

    nperseg = samples
    tau = nperseg/5
    window = signal.windows.exponential(nperseg,tau=tau)

    # Calculates the Power Spectrum Density by Welch's method and its 1st and 2nd deltas
    if subj['mag_diff_f'].values.shape[0] > samples:
        _, psd = signal.welch(subj['mag_diff_f'].values,fs=50,window=window,detrend='linear')
        f_psd_max = np.max(psd)
        f_psd_area = np.sum(psd)
        psd = psd/f_psd_max
        d1psd = np.gradient(psd)
        d2psd = np.gradient(d1psd)
    else:
        psd=d1psd=d2psd=f_psd_max=f_psd_area=None
        
    signal_features = [t_min, t_max, t_mean, t_var, t_skew, t_kurtosis, t_iqr, t_sem, t_mad, f_psd_max, f_psd_area]
    signal_features = np.stack(signal_features)
    
    return psd, d1psd, d2psd, signal_features


def load_subject(subject_id,ids_train,path_train,ids_test,path_test,ignore_test_data):
    ids_train = pd.read_csv(path_train+ids_train)
    psds = list()
    d1psds = list()
    d2psds = list()
    signal_fts = list()
    labels_med = list()
    labels_dys = list()
    labels_tre = list()
    valid_train_files = os.listdir(os.fsencode(path_train))

    for measurement_id in ids_train[ids_train['subject_id'] == subject_id].values[:,0]:
        if os.fsencode(measurement_id+'.csv') in valid_train_files:
            psd, d1psd, d2psd, signal_ft = load_spectrums(measurement_id,path_train)
            if psd is not None:
                psds.append(psd)
                d1psds.append(d1psd)
                d2psds.append(d2psd)
                signal_fts.append(signal_ft)
                labels_med.append(ids_train['on_off'][ids_train['measurement_id'] == measurement_id].values.astype(int)[0])
                labels_dys.append(ids_train['dyskinesia'][ids_train['measurement_id'] == measurement_id].values.astype(int)[0])
                labels_tre.append(ids_train['tremor'][ids_train['measurement_id'] == measurement_id].values.astype(int)[0])
    if not ignore_test_data:
        ids_test = pd.read_csv(path_test+ids_test)
        valid_test_files = os.listdir(os.fsencode(path_test))
        for measurement_id in ids_test[ids_test['subject_id'] == subject_id].values[:,0]:
            if os.fsencode(measurement_id+'.csv') in valid_test_files:
                psd, d1psd, d2psd, signal_ft = load_spectrums(measurement_id,path_test)
                if psd is not None:
                    psds.append(psd)
                    d1psds.append(d1psd)
                    d2psds.append(d2psd)
                    signal_fts.append(signal_ft)
                    labels_med.append(-1)
                    labels_dys.append(-1)
                    labels_tre.append(-1)
    
    psds = np.stack(psds)
    d1psds = np.stack(d1psds)
    d2psds = np.stack(d2psds)
    
    p_value = 0.001
    
    cn_matrix1 = calc_cn_matrix(psds,p_value)
    cn_matrix2 = calc_cn_matrix(d1psds,p_value)
    cn_matrix3 = calc_cn_matrix(d2psds,p_value)
    
    labels_med = np.stack(labels_med).reshape(-1,1)
    labels_dys = np.stack(labels_dys).reshape(-1,1)
    labels_tre = np.stack(labels_tre).reshape(-1,1)
    
    labels = np.hstack((labels_med,labels_dys))
    labels = np.hstack((labels,labels_tre))

    ft_matrix1 = psds
    ft_matrix2 = d1psds
    ft_matrix3 = d2psds
    ft_matrix4 = np.stack(signal_fts)
    
    return cn_matrix1, cn_matrix2, cn_matrix3, ft_matrix1, ft_matrix2, ft_matrix3, ft_matrix4, labels

#-----------------------------------------------------------------------------

args = parser.parse_args()

study = args.study
hdf5_prefix = args.prefix
ignore_test_data = args.ignore_test_data

if study == "CIS":
    path_train="/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
    path_test="/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Test/testing_data/"
    ids_train = "CIS-PD_Training_Data_IDs_Labels.csv"
    ids_test = "cis-pd.CIS-PD_Test_Data_IDs.csv"
    # subjects_list = [1004,1006,1007,1019,1020,1023,1032,1034,1038,1043,1046,1048,1049,1051,1044,1039] #1051,1044,1039

if study == "REAL":
    path_train="/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/REAL/Train/training_data/smartwatch_accelerometer/"
    path_test="/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/REAL/Test/testing_data/smartwatch_accelerometer/"
    ids_train = "REAL-PD_Training_Data_IDs_Labels.csv"
    ids_test = "real-pd.REAL-PD_Test_Data_IDs.csv"
    # subjects_list = ['hbv012','hbv017', 'hbv051',  'hbv077', 'hbv043', 'hbv014', 'hbv018', 'hbv013', 'hbv022', 'hbv023', 'hbv038','hbv054']


if __name__ == '__main__':

    subjects_ids = np.unique(pd.read_csv(path_train+ids_train,usecols=[1]).values).tolist()
    
    f = hdf5_handler(path_train+hdf5_prefix+'.hdf5','a')

    print('Loading data for study {} at {}.hdf5'.format(study,hdf5_prefix))

    for subject_id in subjects_ids:
        print('Loading subject '+str(subject_id))
        data = load_subject(subject_id,ids_train,path_train,ids_test,path_test,ignore_test_data)
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

    


