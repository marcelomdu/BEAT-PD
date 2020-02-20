import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft
from scipy import signal
from scipy.ndimage import gaussian_filter1d as gauss


def calc_fft(x):
    fft_x = fft(x)
    fft_x = abs(fft_x/max(abs(fft_x)))
    return fft_x

def load_subj(x,act):
    subj = pd.read_csv("sub_"+str(x)+"_"+act+".csv",usecols=(10,11,12),nrows=200)
    subj['mag'] = subj.pow(2).sum(1).values
    subj['mag_filt'] = gauss(subj['mag'].values,sigma=0.1)
    subj['fft'] = calc_fft(subj['mag'].values)
    return subj

#-----------------------------------------------------------------------------
    
subjects = {1,3,5,7,9,11,13}
data_walk = list()
data_jog = list()

for x in subjects:
    data_walk.append(load_subj(x,"walk"))
    data_jog.append(load_subj(x,"jog"))

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