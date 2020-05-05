import contextlib
import h5py
import pandas as pd
import numpy as np
from scipy import signal
from numba import njit
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from spectrum import pburg

def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename.encode(), fapl=propfaid)) as fid:
        return h5py.File(fid, mode)

@njit
def calc_mag_diff(x):
    mag_diff = np.zeros(x.shape[0]-2)
    for i in range(1,x.shape[0]-1):
        diff = x[i,:]-x[i-1,:]
        mag_diff[i-1] = np.sqrt(np.power(diff[0],2)+np.power(diff[1],2)+np.power(diff[2],2))
    mag_diff = np.concatenate((mag_diff,np.zeros(2)))
    
    return mag_diff


def load_spectrums(x,folder,interval=4,lf=3,hf=8,th=0.4):
    samples = interval*50 # interval in seconds
    subj = pd.read_csv(folder+x+".csv",usecols=(1,2,3))
    sos = signal.butter(2,[1,20],btype='bandpass',fs=50,output='sos')
    # PCA
    pca = PCA(n_components=1)
    subj['pca_axis'] = pca.fit_transform(subj.values[:,0:3])
    subj['filtered_pca_axis'] = signal.sosfilt(sos,subj['pca_axis'].values)
    # Mag diff
    subj['mag_diff'] = calc_mag_diff(subj.values[:,0:3])
    subj['filtered_mag_diff'] = signal.sosfilt(sos,subj['mag_diff'].values)
    # Threshold frequencies and relative peak power
    lf = int(lf*interval)
    hf = int(hf*interval)
    
    psds_pca = list()
    psds_mag = list()
    wlabels_pca = list()
    wlabels_mag = list()
    fpeaks_pca = list()
    fpeaks_mag = list()
    nperseg = samples
    # tau = nperseg/5
    # window = signal.windows.exponential(nperseg,tau=tau)
    window = 'hann'
    
    for i in range(0,subj.values.shape[0],samples):
        sig1 = subj['filtered_pca_axis'].values[i:i+samples]
        sig2 = subj['filtered_mag_diff'].values[i:i+samples]
        if sig1.shape[0] == samples:
            # PCA PSDs
            freq, psd_pca = signal.welch(sig1,fs=50,window=window,detrend='linear',nperseg=nperseg)
            argmax_pca = np.argmax(psd_pca[lf:hf])+lf
            fpeaks_pca.append(np.around(freq[argmax_pca],decimals=2))
            pratio_pca = np.sum(psd_pca[argmax_pca-1:argmax_pca+1])/np.sum(psd_pca[lf:hf])
            if pratio_pca > th:
                wlabels_pca.append(1)
            else:
                wlabels_pca.append(0)
            psd_pca = psd_pca[lf-3:hf+3]#/np.max(psd_pca[lf-3:hf+3])
            psds_pca.append(psd_pca)

            # Mag diff PSDs
            freq, psd_mag = signal.welch(sig2,fs=50,window=window,detrend='linear',nperseg=nperseg)
            argmax_mag = np.argmax(psd_mag[lf:hf])+lf
            fpeaks_mag.append(np.around(freq[argmax_mag],decimals=2))
            pratio_mag = np.sum(psd_mag[argmax_mag-1:argmax_mag+1])/np.sum(psd_mag[lf:hf])
            if pratio_mag > th:
                wlabels_mag.append(1)
            else:
                wlabels_mag.append(0)
            psd_mag = psd_mag[lf-3:hf+3]#/np.max(ps_mag[lf-3:hf+3])
            psds_mag.append(psd_mag)

    wlabels_pca = np.stack(wlabels_pca[:-1])
    fpeaks_pca = np.stack(fpeaks_pca[:-1])
    psds_pca = np.stack(psds_pca[:-1])
    
    wlabels_mag = np.stack(wlabels_mag[:-1])
    fpeaks_mag = np.stack(fpeaks_mag[:-1])
    psds_mag = np.stack(psds_mag[:-1])

    wlabels = [wlabels_pca,wlabels_mag]
    fpeaks = [fpeaks_pca,fpeaks_mag]
    psds = [psds_pca,psds_mag]
   
    x = freq[lf-3:hf+3]
    y_w = np.zeros(x.shape)
    y_w[3]=y_w[hf-lf+3]=1
    
    # for i in range(0,common_tpsd1.shape[0]):
    #     plt.figure(i)
    #     plt.title("AND")
    #     plt.plot(x,common_tpsd1[i,:])
    #     plt.plot(x,common_tpsd2[i,:])
    #     plt.plot(x,y_w)
    # for j in range(0,xor_tpsd1.shape[0]):
    #     plt.figure(j+i+1)
    #     plt.title("XOR")
    #     plt.plot(x,xor_tpsd1[j,:])
    #     plt.plot(x,xor_tpsd2[j,:])
    #     plt.plot(x,y_w)
    # plt.show()
    
    return wlabels,fpeaks,psds,x,y_w

def load_measurement(x,folder,interval=10):
    samples = interval*50 # ten seconds interval from a 50 Hz sample rate
    subj = pd.read_csv(folder+x+".csv",usecols=(1,2,3))
    subj['mag'] = subj.pow(2).sum(1).values
    subj['mag'] = np.sqrt(subj['mag'].values)
    subj['mag_diff'] = calc_mag_diff(subj.values)
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
    subject_psds = list()
    subject_spects = list()
    measurements_labels_medication = list()
    measurements_labels_dyskinesia = list()
    measurements_labels_tremor = list()
    measurements_labels = dict()
    for measurement_id in ids_file[ids_file['subject_id'] == subject_id].values[:,0]:
        subject_psd, subject_spect = load_spectrums(measurement_id,folder)
        subject_psds.append(subject_psd)
        subject_spects.append(subject_spect)
        measurements_labels_medication.append(ids_file['on_off'][ids_file['measurement_id'] == measurement_id].values.astype(int))
        measurements_labels_dyskinesia.append(ids_file['dyskinesia'][ids_file['measurement_id'] == measurement_id].values.astype(int))
        measurements_labels_tremor.append(ids_file['tremor'][ids_file['measurement_id'] == measurement_id].values.astype(int))
    measurements_labels_medication = np.stack(measurements_labels_medication)
    measurements_labels_dyskinesia = np.stack(measurements_labels_dyskinesia)
    measurements_labels_tremor = np.stack(measurements_labels_tremor)
    measurements_labels = np.hstack((measurements_labels_medication,measurements_labels_dyskinesia))
    measurements_labels = np.hstack((measurements_labels,measurements_labels_tremor))
    
    return subject_psds, subject_spects, measurements_labels

#-----------------------------------------------------------------------------
    
if __name__ == '__main__':

    subjects_ids = [1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051]
    ids_file = "CIS-PD_Training_Data_IDs_Labels.csv"
    folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
    # f = hdf5_handler(folder+'training_data_PSD.hdf5','a')

    for subject_id in subjects_ids:
        print('Loading subject '+str(subject_id))
        # subj = f.create_group(str(subject_id))
        # measurements = subj.create_group('measurements')
        data = load_subject(subject_id,ids_file,folder)
        for i in range(0,len(data[0])):
            if i < 100:
                if i < 10:
                    n = '00'+str(i)
                else:
                    n = '0'+str(i)
            else:
                n = str(i)
        #     measurements.create_dataset('PSD'+n,data=data[0][i])
        #     measurements.create_dataset('Spect'+n, data=data[1][i])
        # subj.create_dataset('labels', data=data[2])        

    
    print('Prepare data done!')