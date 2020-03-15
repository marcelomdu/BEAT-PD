import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import scipy.cluster.hierarchy as sch
from numba import njit

def load_subj(x,folder):
    sos = signal.butter(10, [2,20],btype='bandpass',fs=50,output='sos')
    interval = 10
    samples = interval*50 # ten seconds interval
    subj = pd.read_csv(folder+x+".csv",usecols=(1,2,3))
    subj['X'] = signal.sosfilt(sos, subj['X'].values)
    subj['Y'] = signal.sosfilt(sos, subj['Y'].values)
    subj['Z'] = signal.sosfilt(sos, subj['Z'].values)
    subj['mag_diff'] = calc_mag_diff(subj.values)
    psd = list()
    window = 'hann'
    for i in range(0,subj.values.shape[0],samples):
        fs, ps = signal.welch(subj['mag_diff'].values[i:i+samples-1],fs=50,window=window,detrend='linear',nperseg=1024)
        ps = ps/np.max(ps) # Data normalization
        # ps = 1/(-np.log10(ps))
        psd.append(ps)
    subj['mag_SMA'] = subj.iloc[:,3].rolling(window=2).mean()
    n_psd = len(psd)
    psd = np.concatenate((psd[:-1])).reshape(len(psd)-1,int((samples/2)))
    sum_psd = np.sum(psd,axis=0)/n_psd
    return subj, psd, n_psd, sum_psd

@njit
def calc_mag_diff(x):
    mag_diff = np.zeros(x.shape[0]-2)
    for i in range(1,x.shape[0]-1):
        diff = x[i,:]-x[i-1,:]
        mag_diff[i-1] = np.sqrt(np.power(diff[0],2)+np.power(diff[1],2)+np.power(diff[2],2))-1
    mag_diff = np.concatenate((mag_diff,np.zeros(2)))
    return mag_diff

#-----------------------------------------------------------------------------

test_files = pd.read_csv("test_files.txt",header=None,sep="\t").values
data = list()
folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"

for x in test_files[:,0]:
    data.append(load_subj(x, folder))


t = len(data[0][1])

for j in range(3,4):
    for i in range(0,4):
        plt.figure(i)
        plt.figure(figsize=(5, 4))
        plt.title('PSD: power spectral density '+str(j))
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.tight_layout()
        plt.plot(data[j][1][i])
        #plt.semilogx(data[j][1][i])


for subj in range(2,3):
    plt.figure(1)
    plt.title('Subject:'+str(subj)+' Tremor: '+str(test_files[subj,2]))
    plt.plot(data[subj][3])
    #plt.semilogx(1/(-np.log10(data[subj][3])))
    #plt.imshow(data[subj][1], cmap='viridis')
    plt.show()
    plt.figure(2)
    plt.title('Subject:'+str(subj)+' Tremor: '+str(test_files[subj,2]))
    plt.semilogx(data[subj][3])
    #plt.semilogx(1/(-np.log10(data[subj][3])))
    #plt.imshow(data[subj][1], cmap='viridis')
    plt.show()
    plt.figure(3)
    plt.title('Subject:'+str(subj)+' Tremor: '+str(test_files[subj,2]))
    plt.loglog(data[subj][3])
    #plt.semilogx(1/(-np.log10(data[subj][3])))
    #plt.imshow(data[subj][1], cmap='viridis')
    plt.show()


for subj in range(0,10):
    plt.figure(subj)
    plt.title('Subject:'+str(subj)+' Tremor: '+str(test_files[subj,4]))
    plt.imshow(data[subj][1], cmap='viridis')
    plt.show()


















