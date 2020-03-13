import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import scipy.cluster.hierarchy as sch
from numba import njit
from itertools import compress
from audioop import rms


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
    rms = list()
    window = 'hann'
    fs, mag_diff_psd = signal.welch(subj['mag_diff'].values[:],fs=50,window=window,detrend='linear',nperseg=1024)
    for i in range(0,subj.values.shape[0],samples):
        fs, ps = signal.welch(subj['mag_diff'].values[i:i+samples-1],fs=50,window=window,detrend='linear',nperseg=1024)
        ps = ps/np.max(ps) # Data normalization
        i_rms = rms(subj['mag_diff'].values[i:i+samples-1],4)
        # ps = 1/(-np.log10(ps))
        psd.append(ps)
        
    subj['mag_SMA'] = subj.iloc[:,3].rolling(window=2).mean()
    n_psd = len(psd)
    psd = np.concatenate((psd[:-1])).reshape(len(psd)-1,int((samples/2)))
    sum_psd = np.sum(psd,axis=0)/n_psd
    return subj, psd, n_psd, sum_psd, mag_diff_psd

@njit
def calc_mag_diff(x):
    mag_diff = np.zeros(x.shape[0]-2)
    for i in range(1,x.shape[0]-1):
        diff = x[i,:]-x[i-1,:]
        mag_diff[i-1] = np.sqrt(np.power(diff[0],2)+np.power(diff[1],2)+np.power(diff[2],2))
    mag_diff = np.concatenate((mag_diff,np.zeros(2)))
    return mag_diff

#-----------------------------------------------------------------------------

#test_files = pd.read_csv("test_files.txt",header=None,sep="\t").values
data = list()
folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
files = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/CIS-PD_Training_Data_IDs_Labels.csv"

subject = 1046

data_files = pd.read_csv(files)

test_files = data_files[data_files['subject_id']==subject].values

for x in test_files[:,0]:
    data.append(load_subj(x, folder))

for i in len(data):
    if 


data0 = list(compress(data,test_files[:,4]==0))
data1 = list(compress(data,test_files[:,4]==1))
data2 = list(compress(data,test_files[:,4]==2))
data3 = list(compress(data,test_files[:,4]==3))
data4 = list(compress(data,test_files[:,4]==4))

tremor_all = list()
tremor0 = list()
tremor1 = list()
tremor2 = list()
tremor3 = list()
tremor4 = list()

for i in range(0,len(data)):
    tremor_all.append(data[i][4])

for i in range(0,len(data0)):
    tremor0.append(data0[i][4])

for i in range(0,len(data1)):
    tremor1.append(data1[i][4])

for i in range(0,len(data2)):
    tremor2.append(data2[i][4])

for i in range(0,len(data3)):
    tremor3.append(data3[i][4])
    
for i in range(0,len(data4)):
    tremor4.append(data4[i][4])

if len(data0)>0:    
    tremor0_mean = np.mean(np.vstack(tremor0),axis=0)#/len(data0)
    #tremor0_mean = tremor0_mean/np.max(tremor0_mean)
if len(data1)>0:   
    tremor1_mean = np.mean(np.vstack(tremor1),axis=0)#/len(data1)
    #tremor1_mean = tremor1_mean/np.max(tremor1_mean)#-tremor0_mean
if len(data2)>0:   
    tremor2_mean = np.mean(np.vstack(tremor2),axis=0)#/len(data2)
    #tremor2_mean = tremor2_mean/np.max(tremor2_mean)#-tremor0_mean
if len(data3)>0:   
    tremor3_mean = np.mean(np.vstack(tremor3),axis=0)#/len(data3)
    #tremor3_mean = tremor3_mean/np.max(tremor3_mean)#-tremor0_mean
if len(data4)>0:
    tremor4_mean = np.mean(np.vstack(tremor4),axis=0)#/len(data4)
    #tremor4_mean = tremor4_mean/np.max(tremor4_mean)#-tremor0_mean

plt.figure(0)
plt.loglog(tremor0_mean)
plt.figure(0)
plt.loglog(tremor1_mean)
plt.figure(0)
plt.loglog(tremor2_mean)
plt.figure(0)
plt.loglog(tremor3_mean)
plt.figure(0)
plt.loglog(tremor4_mean)

plt.figure(1)
plt.semilogx(tremor0_mean)
plt.figure(1)
plt.semilogx(tremor1_mean)
plt.figure(1)
plt.semilogx(tremor2_mean)
plt.figure(1)
plt.semilogx(tremor3_mean)
plt.figure(1)
plt.semilogx(tremor4_mean)





plt.figure(1)
plt.loglog(data[0][4])
plt.figure(2)
plt.loglog(data[0][3])

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


















