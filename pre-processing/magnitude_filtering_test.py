import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from numba import njit


def load_subj(x,folder):
    sos = signal.butter(10,4,btype='high',fs=50,output='sos')
    interval = 5 # in seconds
    samples = interval*50
    subj = pd.read_csv(folder+x+".csv",usecols=(1,2,3))
    subj['mag_diff'] = calc_mag_diff(subj.values[:,0:3])
    subj['mag_diff_f'] = signal.sosfilt(sos,subj['mag_diff'].values)
    psd = list()
    # window = 'hann'
    nperseg = samples
    tau = nperseg/5
    window = signal.windows.exponential(nperseg,tau=tau)
    # Calculates the Power Spectrum Density by Welch's method and its 1st and 2nd deltas
    for i in range(0,subj.values.shape[0],samples):
        s = subj['mag_diff_f'].values[i:i+samples]
        if s.shape[0] == samples:
            _, ps = signal.welch(s,fs=50,window=window,detrend='linear',nperseg=nperseg)
            # Normalize data to the [0,1] interval
            ps = ps/np.max(ps)
            # Take only frequencies above 3.5 Hz 
            psd.append(ps[35:])
    n_psd = len(psd)
    psd = np.vstack(psd[:-1])
    d1psd = (np.insert(psd,0,0,axis=0)-np.insert(psd,psd.shape[0],0,axis=0))[1:-1,:]
    d2psd = (np.insert(d1psd,0,0,axis=0)-np.insert(d1psd,d1psd.shape[0],0,axis=0))[1:-1,:]
    # Arrange time intervals from lowest to highest peak frequency
    d1psd = d1psd[np.argmax(psd[1:-1],axis=1).argsort()]
    d2psd = d2psd[np.argmax(psd[2:-2],axis=1).argsort()]
    psd = psd[np.argmax(psd,axis=1).argsort()]
    # Stack PSD and its deltas
    st_psd = np.stack((psd[4:,:],d1psd[2:,:],d2psd))
    # Calculates the Spectrogram and its 1st and 2nd deltas
    spect = signal.spectrogram(subj['mag_diff_f'].values,fs=50,window=window,detrend='linear',nperseg=nperseg)
    spect = (spect[2]/np.max(spect[2],axis=0)).T[:,35:]
    d1spect = (np.insert(spect,0,0,axis=0)-np.insert(spect,spect.shape[0],0,axis=0))[1:-1,:]
    d2spect = (np.insert(d1spect,0,0,axis=0)-np.insert(d1spect,d1spect.shape[0],0,axis=0))[1:-1,:]
    # Arrange time intervals from lowest to highest peak frequency
    d1spect = d1spect[np.argmax(spect[1:-1],axis=1).argsort()]
    d2spect = d2spect[np.argmax(spect[2:-2],axis=1).argsort()]
    spect = spect[np.argmax(spect,axis=1).argsort()]
    # Stack PSD and its deltas
    st_spect = np.stack((spect[4:,:],d1spect[2:,:],d2spect))
    
    return subj, n_psd, st_psd, st_spect

@njit
def calc_mag_diff(x):
    mag_diff = np.zeros(x.shape[0]-2)
    for i in range(1,x.shape[0]-1):
        diff = x[i,:]-x[i-1,:]
        mag_diff[i-1] = np.sqrt(np.power(diff[0],2)+np.power(diff[1],2)+np.power(diff[2],2))
    mag_diff = np.concatenate((mag_diff,np.zeros(2)))
    return mag_diff

#-----------------------------------------------------------------------------

if __name__ == '__main__':

    #test_files = pd.read_csv("test_files.txt",header=None,sep="\t").values
    data = list()
    folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
    files = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/CIS-PD_Training_Data_IDs_Labels.csv"

    subject = 1004

    data_files = pd.read_csv(files)

    test_files = pd.DataFrame(data_files[data_files['subject_id']==subject].values)

    for x in test_files.values[:,0]:
        data.append(load_subj(x, folder))

    labels = {}

    labels[0] = test_files[test_files[4]==0].index.tolist()
    labels[1] = test_files[test_files[4]==1].index.tolist()
    labels[2] = test_files[test_files[4]==2].index.tolist()
    labels[3] = test_files[test_files[4]==3].index.tolist()
    labels[4] = test_files[test_files[4]==4].index.tolist()

    n = 4

    for i in range(0,5):
        if len(labels[i])>0:
            for j in labels[i][:n]:
                plt.figure(str(j))
                plt.title(str(i))
                plt.subplot(141)
                plt.imshow(data[int(j)][2][0,:,:], cmap='viridis')
                plt.subplot(142)
                plt.imshow(data[int(j)][2][1,:,:], cmap='viridis')
                plt.subplot(143)
                plt.imshow(data[int(j)][2][2,:,:], cmap='viridis')

