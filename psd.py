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
    mag = np.zeros(subj.values.shape[0]-2)
    for i in range(1,subj.values.shape[0]-1):
        diff = subj.values[i,:]-subj.values[i-1,:]
        mag[i-1] = math.sqrt(pow(diff[0],2)+pow(diff[1],2)+pow(diff[2],2))
    # subj['mag'] = subj.pow(2).sum(1).values
    # subj['mag_filt'] = gauss(subj['mag'].values,sigma=0.1)
    # subj['fft'] = calc_fft(subj['mag'].values)
    return mag

#-----------------------------------------------------------------------------
    
test_files = pd.read_csv("test_files.txt",header=None,sep="\t").values
data = list()
folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"

for x in test_files[:,0]:
    data.append(load_subj(x,folder))

for i in range(0,len(data_jog)):
    plt.plot(gauss(data_jog[i].values[:,4],sigma=0.5))
    

s = 6

sig1 = data_jog[s].values[:,3]
sig2 = data_walk[s].values[:,3]

freqs, times, spectrogram = signal.spectrogram(sig2)

plt.figure(figsize=(5, 4))
plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
plt.title('Spectrogram')
plt.ylabel('Frequency band')
plt.xlabel('Time window')
plt.tight_layout()

freqs, psd = signal.welch(sig1)

plt.figure(figsize=(5, 4))
plt.semilogx(freqs, psd)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()