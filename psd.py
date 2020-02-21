import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft
from scipy import signal
from scipy.ndimage import gaussian_filter1d as gauss
import math


def calc_fft(x):
    fft_x = fft(x)
    fft_x = abs(fft_x/max(abs(fft_x)))
    return fft_x

def load_subj(x,folder):
    subj = pd.read_csv(folder+x+".csv",usecols=(1,2,3))
    # mag_diff = np.zeros(subj.values.shape[0]-2)
    # for i in range(1,subj.values.shape[0]-1):
    #     diff = subj.values[i,:]-subj.values[i-1,:]
    #     mag_diff[i-1] = math.sqrt(pow(diff[0],2)+pow(diff[1],2)+pow(diff[2],2))
    # x = np.array([0,1,2,3,4])*math.pi/4
    # f = np.sin(x)
    subj['mag'] = subj.pow(2).sum(1).values
    subj['mag_SMA'] = subj.iloc[:,3].rolling(window=2).mean()
    # subj['mag_conv'] = np.convolve(subj['mag'].values,f)[:-4]
    # subj['mag_diff'] = np.concatenate((mag_diff,[0,0]))
    return subj

#-----------------------------------------------------------------------------
    
test_files = pd.read_csv("test_files.txt",header=None,sep="\t").values
data = list()
folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"

for x in test_files[:,0]:
    data.append(load_subj(x,folder))

subs0 = {1}
subs1 = {0,2,4,5}
subs2 = {6,7}
subs3 = {3,8,9}

group = subs3

for i in group:
    plt.plot(data[i].values[:,3])
    
    
freqs = list()
psd = list()    

plt.figure(figsize=(5, 4))
for i in group:
    fs, ps = signal.welch(data[i].values[1:,4],fs=1/0.02)
    ps = ps/max(ps)
    freqs.append(fs)
    psd.append(ps)
    #plt.plot(fs, ps)
    plt.semilogx(fs, ps)
    
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()