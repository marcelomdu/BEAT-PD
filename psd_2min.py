import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import scipy.cluster.hierarchy as sch
from numba import njit

def load_subj(x,folder):
    samples = 100*30 # one minute interval
    subj = pd.read_csv(folder+x+".csv",usecols=(1,2,3))
    subj['mag_diff'] = calc_mag_diff(subj.values)
    subj['mag'] = subj.pow(2).sum(1).values
    subj['mag'] = np.sqrt(subj['mag'].values)
    psd = list()
    window = 'hann'
    for i in range(0,subj.values.shape[0],samples):
        fs, ps = signal.welch(subj['mag_diff'].values[i:i+samples-1],fs=50,window=window)
        ps = ps/np.max(ps)
        psd.append(ps)
    subj['mag_SMA'] = subj.iloc[:,3].rolling(window=2).mean()
    psd = np.concatenate((psd[:-1])).reshape(len(psd)-1,129)
    return subj, psd

@njit
def calc_mag_diff(x):
    mag_diff = np.zeros(x.shape[0]-2)
    for i in range(1,x.shape[0]-1):
        diff = x[i,:]-x[i-1,:]
        mag_diff[i-1] = np.sqrt(np.power(diff[0],2)+np.power(diff[1],2)+np.power(diff[2],2))
    mag_diff = np.concatenate((mag_diff,np.zeros(2)))
    return mag_diff

#-----------------------------------------------------------------------------

test_files = pd.read_csv("test_files.txt",header=None,sep="\t").values
data = list()
folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"

for x in test_files[:10,0]:
    data.append(load_subj(x,folder))


t = len(data[0][1])

for j in range(0,2):
    for i in range(0,4):
        plt.figure(i)
        plt.figure(figsize=(5, 4))
        plt.title('PSD: power spectral density '+str(j))
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.tight_layout()
        #plt.plot(data[j][1][i])
        plt.semilogx(data[j][1][i])


subj = 4
a = 200
b = 300

plt.imshow(data[subj][1], cmap='viridis')
plt.title('Subject:'+str(subj)+', Interval: ['+str(a)+':'+str(b)+']')
plt.show()

















X = data[3][1].T[:,:]

dendrogram = sch.dendrogram(sch.linkage(X, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean distances')
plt.show()

plt.plot(data[0][0]['mag_diff'].values)
    
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