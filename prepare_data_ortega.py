#%%
import contextlib
import h5py
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, skew, kurtosis

def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename.encode(), fapl=propfaid)) as fid:
        return h5py.File(fid, mode)

def calc_psds(Xw,Yw,Zw,Aw,pcaw,window,nperseg,lf,hf,all_psds=False):
    if not all_psds:
        # PSDs PCA
        freqs, psd_pca = signal.welch(pcaw,fs=50,window=window,detrend='linear',nperseg=nperseg)
        argmax_pca = np.argmax(psd_pca[lf:hf])+lf
        fpeak_pca = np.around(freqs[argmax_pca],decimals=2)
        pratio_pca = np.sum(psd_pca[argmax_pca-1:argmax_pca+1])/np.sum(psd_pca[lf:hf])
        psd_pca = psd_pca[lf-3:hf+3]#/np.max(psd_pca[lf-3:hf+3])        
        # Outputs
        psds = psd_pca
        fpeaks = fpeak_pca
    else: 
        # Currently unused
        # PSDs PCA
        freqs, psd_pca = signal.welch(pcaw,fs=50,window=window,detrend='linear',nperseg=nperseg)
        argmax_pca = np.argmax(psd_pca[lf:hf])+lf
        fpeak_pca = np.around(freqs[argmax_pca],decimals=2)
        pratio_pca = np.sum(psd_pca[argmax_pca-1:argmax_pca+1])/np.sum(psd_pca[lf:hf])
        psd_pca = psd_pca[lf-3:hf+3]#/np.max(psd_pca[lf-3:hf+3])
        # PSDs Axis
        freqs, psd_x = signal.welch(Xw,fs=50,window=window,detrend='linear',nperseg=nperseg)
        argmax_x = np.argmax(psd_x[lf:hf])+lf
        fpeak_x = np.around(freqs[argmax_x],decimals=2)
        psd_x = psd_x[lf-3:lf+3]
        freqs, psd_y = signal.welch(Yw,fs=50,window=window,detrend='linear',nperseg=nperseg)
        argmax_y = np.argmax(psd_y[lf:hf])+lf
        fpeak_y = np.around(freqs[argmax_y],decimals=2)
        psd_y = psd_y[lf-3:lf+3]
        freqs, psd_z = signal.welch(Zw,fs=50,window=window,detrend='linear',nperseg=nperseg)
        argmax_z = np.argmax(psd_z[lf:hf])+lf
        fpeak_z = np.around(freqs[argmax_z],decimals=2)
        psd_z = psd_z[lf-3:lf+3]
        freqs, psd_a = signal.welch(Aw,fs=50,window=window,detrend='linear',nperseg=nperseg)
        argmax_a = np.argmax(psd_a[lf:hf])+lf
        fpeak_a = np.around(freqs[argmax_a],decimals=2)
        psd_a = psd_a[lf-3:lf+3]
        # Outputs
        psds = [psd_x,psd_y,psd_z,psd_a,psd_pca]
        fpeaks = [fpeak_x,fpeak_y,fpeak_z,fpeak_a,fpeak_pca]   
    return psds, fpeaks, pratio_pca, freqs


def calc_features(Xw,Yw,Zw,Aw,dXw,dYw,dZw,dAw,pcaw,dpcaw,samples):
    # Features calculation
    # Extracts statistical features from the time series
    wf = list()
    # Mean (indices 0-9)
    wf.append(np.mean(Xw))
    wf.append(np.mean(Yw))
    wf.append(np.mean(Zw))
    wf.append(np.mean(Aw))
    wf.append(np.mean(pcaw))
    wf.append(np.mean(dXw))
    wf.append(np.mean(dYw))
    wf.append(np.mean(dZw))
    wf.append(np.mean(dAw))
    wf.append(np.mean(dpcaw))
    # Standard Deviation (indices 10-19)
    wf.append(np.std(Xw))
    wf.append(np.std(Yw))
    wf.append(np.std(Zw))
    wf.append(np.std(Aw))
    wf.append(np.std(pcaw))
    wf.append(np.std(dXw))
    wf.append(np.std(dYw))
    wf.append(np.std(dZw))
    wf.append(np.std(dAw))
    wf.append(np.std(dpcaw))
    # Signal Magnitude Area (indices 20-25)
    wf.append(np.divide(np.sum(np.add(np.abs(Xw-np.mean(Xw)),np.add(np.abs(Yw-np.mean(Yw)),np.abs(Zw-np.mean(Zw))))),samples))
    wf.append(np.divide(np.sum(np.abs(Aw-np.mean(Aw))),samples))
    wf.append(np.divide(np.sum(np.abs(Aw-np.mean(pcaw))),samples))
    wf.append(np.divide(np.sum(np.add(np.abs(dXw-np.mean(dXw)),np.add(np.abs(dYw-np.mean(dYw)),np.abs(dZw-np.mean(dZw))))),samples))
    wf.append(np.divide(np.sum(np.abs(Aw-np.mean(dAw))),samples))
    wf.append(np.divide(np.sum(np.abs(Aw-np.mean(dpcaw))),samples))
    # Entropy (indices 26-35) Commented due to division by zero problem
    # wf.append(np.sum(np.multiply(np.abs(Xw-np.mean(Xw)),np.log10(np.abs(Xw-np.mean(Xw))))))
    # wf.append(np.sum(np.multiply(np.abs(Yw-np.mean(Yw)),np.log10(np.abs(Yw-np.mean(Yw))))))
    # wf.append(np.sum(np.multiply(np.abs(Zw-np.mean(Zw)),np.log10(np.abs(Zw-np.mean(Zw))))))
    # wf.append(np.sum(np.multiply(np.abs(Aw-np.mean(Aw)),np.log10(np.abs(Aw-np.mean(Aw))))))
    # wf.append(np.sum(np.multiply(np.abs(pcaw-np.mean(pcaw)),np.log10(np.abs(pcaw-np.mean(pcaw))))))
    # wf.append(np.sum(np.multiply(np.abs(dXw-np.mean(dXw)),np.log10(np.abs(dXw-np.mean(dXw))))))
    # wf.append(np.sum(np.multiply(np.abs(dYw-np.mean(dYw)),np.log10(np.abs(dYw-np.mean(dYw))))))
    # wf.append(np.sum(np.multiply(np.abs(dZw-np.mean(dZw)),np.log10(np.abs(dZw-np.mean(dZw))))))
    # wf.append(np.sum(np.multiply(np.abs(dAw-np.mean(dAw)),np.log10(np.abs(dAw-np.mean(dAw))))))
    # wf.append(np.sum(np.multiply(np.abs(dpcaw-np.mean(dpcaw)),np.log10(np.abs(dpcaw-np.mean(dpcaw))))))
    # Correlation (indices 36-41)
    xyw,_= pearsonr(Xw,Yw)
    xzw,_= pearsonr(Xw,Zw)
    yzw,_= pearsonr(Yw,Zw)
    dxdyw,_= pearsonr(dXw,dYw)
    dxdzw,_= pearsonr(dXw,dZw)
    dydzw,_= pearsonr(dYw,dZw)
    wf.append(xyw)
    wf.append(xzw)
    wf.append(yzw)
    wf.append(dxdyw)
    wf.append(dxdzw)
    wf.append(dydzw)
    # Skewness (indices 42-51)
    wf.append(skew(Xw))
    wf.append(skew(Yw))
    wf.append(skew(Zw))
    wf.append(skew(Aw))
    wf.append(skew(pcaw))
    wf.append(skew(dXw))
    wf.append(skew(dYw))
    wf.append(skew(dZw))
    wf.append(skew(dAw))
    wf.append(skew(dpcaw))
    # Kurtosis (indices 52-61)
    wf.append(kurtosis(Xw))
    wf.append(kurtosis(Yw))
    wf.append(kurtosis(Zw))
    wf.append(kurtosis(Aw))
    wf.append(kurtosis(pcaw))
    wf.append(kurtosis(dXw))
    wf.append(kurtosis(dYw))
    wf.append(kurtosis(dZw))
    wf.append(kurtosis(dAw))
    wf.append(kurtosis(dpcaw))
    # RMS (indices 62-71)
    wf.append(np.sqrt(np.divide(np.sum(np.power(Xw,2)),samples)))
    wf.append(np.sqrt(np.divide(np.sum(np.power(Yw,2)),samples)))
    wf.append(np.sqrt(np.divide(np.sum(np.power(Zw,2)),samples)))
    wf.append(np.sqrt(np.divide(np.sum(np.power(Aw,2)),samples)))
    wf.append(np.sqrt(np.divide(np.sum(np.power(pcaw,2)),samples)))
    wf.append(np.sqrt(np.divide(np.sum(np.power(dXw,2)),samples)))
    wf.append(np.sqrt(np.divide(np.sum(np.power(dYw,2)),samples)))
    wf.append(np.sqrt(np.divide(np.sum(np.power(dZw,2)),samples)))
    wf.append(np.sqrt(np.divide(np.sum(np.power(dAw,2)),samples)))
    wf.append(np.sqrt(np.divide(np.sum(np.power(dpcaw,2)),samples)))
    # Energy (indices 72-81)
    wf.append(np.divide(np.sum(np.power(np.abs(np.fft.fft(Xw)),2)),samples))
    wf.append(np.divide(np.sum(np.power(np.abs(np.fft.fft(Yw)),2)),samples))
    wf.append(np.divide(np.sum(np.power(np.abs(np.fft.fft(Zw)),2)),samples))
    wf.append(np.divide(np.sum(np.power(np.abs(np.fft.fft(Aw)),2)),samples))
    wf.append(np.divide(np.sum(np.power(np.abs(np.fft.fft(pcaw)),2)),samples))
    wf.append(np.divide(np.sum(np.power(np.abs(np.fft.fft(dXw)),2)),samples))
    wf.append(np.divide(np.sum(np.power(np.abs(np.fft.fft(dYw)),2)),samples))
    wf.append(np.divide(np.sum(np.power(np.abs(np.fft.fft(dZw)),2)),samples))
    wf.append(np.divide(np.sum(np.power(np.abs(np.fft.fft(dAw)),2)),samples))
    wf.append(np.divide(np.sum(np.power(np.abs(np.fft.fft(dpcaw)),2)),samples))
    wf = np.stack(wf)
    return wf

def load_spectrums(x,folder,interval=4,overlap=0.2,lf=3,hf=8,th=0.6,all_psds=False,wlabels_only=False):
    samples = interval*50 # interval in seconds
    subj = pd.read_csv(folder+x+".csv")
    if 'x' in subj.columns:
        subj = subj[['x','y','z']]
    elif 'X' in subj.columns:
        subj = subj[['X','Y','Z']]
    # Calculate A as the magnitude of (x,y,z) vector
    subj['A'] = np.sqrt(np.add(np.power(subj.values[:,0],2),np.add(np.power(subj.values[:,1],2),np.power(subj.values[:,2],2))))
    # Median filter
    subj['X_fm'] = signal.medfilt(subj['X'].values,kernel_size=[7])#subj['X']
    subj['Y_fm'] = signal.medfilt(subj['Y'].values,kernel_size=[7])#subj['Y']
    subj['Z_fm'] = signal.medfilt(subj['Z'].values,kernel_size=[7])#subj['Z']
    subj['A_fm'] = signal.medfilt(subj['A'].values,kernel_size=[7])#subj['A']
    # Bandpass Butterworth filter
    sosl = signal.butter(3,[1,20],btype='bandpass',fs=50,output='sos')
    subj['X_fb'] = signal.sosfilt(sosl,subj['X_fm'].values)
    subj['Y_fb'] = signal.sosfilt(sosl,subj['Y_fm'].values)
    subj['Z_fb'] = signal.sosfilt(sosl,subj['Z_fm'].values)
    subj['A_fb'] = signal.sosfilt(sosl,subj['A_fm'].values)
    # PCA for main motion axis extraction (not included in Ortega et al.)
    pca = PCA(n_components=1)
    subj['PCA'] = pca.fit_transform(subj[['X_fm','Y_fm','Z_fm']].values)
    # Jerk vectors calculation
    subj['dX'] = np.gradient(subj['X_fb'].values)
    subj['dY'] = np.gradient(subj['Y_fb'].values)
    subj['dZ'] = np.gradient(subj['Z_fb'].values)
    subj['dA'] = np.gradient(subj['A_fb'].values)
    subj['dPCA'] = np.gradient(subj['PCA'].values)
    # Threshold frequencies for relative peak power calculation
    lf = int(lf*interval)
    hf = int(hf*interval)
    nperseg = samples
    # tau = nperseg/5
    # window = signal.windows.exponential(nperseg,tau=tau)
    window = 'hann'
    psds_x = list()
    psds_y = list()
    psds_z = list()
    psds_a = list()
    psds_pca = list()
    psds = list()
    fpeaks_x = list()
    fpeaks_y = list()
    fpeaks_z = list()
    fpeaks_a = list()
    fpeaks_pca = list()
    fpeaks = list()
    wfeatures = list()
    wlabels_pca = list()
    # Divide time series in 4 s windows, extracts statistical features (Ortega-Anderez,2018) and classify as tremor or non tremor windows (Luft,2019)
    for i in range(0,subj.values.shape[0],int(samples)):
        Xw = subj['X_fb'].values[i:i+samples]
        Yw = subj['Y_fb'].values[i:i+samples]
        Zw = subj['Z_fb'].values[i:i+samples]
        Aw = subj['A_fb'].values[i:i+samples]
        pcaw = subj['PCA'].values[i:i+samples]
        dXw = subj['dX'].values[i:i+samples]
        dYw = subj['dY'].values[i:i+samples]
        dZw = subj['dZ'].values[i:i+samples]
        dAw = subj['dA'].values[i:i+samples]
        dpcaw = subj['dPCA'].values[i:i+samples]
        if pcaw.shape[0] == samples:
            psdsw, fpeaksw, pratio_pca, freqs = calc_psds(Xw,Yw,Zw,Aw,pcaw,window,nperseg,lf,hf,all_psds)
            if pratio_pca > th:
                wlabels_pca.append(1)
            else:
                wlabels_pca.append(0)
            # time window statistical features            
            wf = calc_features(Xw,Yw,Zw,Aw,dXw,dYw,dZw,dAw,pcaw,dpcaw,samples)
            wfeatures.append(wf)
            if all_psds:
                psds_x.append(psdsw[0])
                psds_y.append(psdsw[1])
                psds_z.append(psdsw[2])
                psds_a.append(psdsw[3])
                fpeaks_x.append(fpeaksw[0])
                fpeaks_y.append(fpeaksw[1])
                fpeaks_z.append(fpeaksw[2])
                fpeaks_a.append(fpeaksw[3])
                psds_pca.append(psdsw[4])
                fpeaks_pca.append(fpeaksw[4])
            else:
                psds_pca.append(psdsw)
                fpeaks_pca.append(fpeaksw)               
    freqs = freqs[lf:hf]
    psds.append(np.stack(psds_pca[:-1]))
    fpeaks.append(np.stack(fpeaks_pca[:-1]))
    wlabels = np.stack(wlabels_pca[:-1])
    wfeatures = np.stack(wfeatures[:-1])    
    if all_psds:
        psds.append(np.stack(psds_x[:-1]))
        psds.append(np.stack(psds_y[:-1]))
        psds.append(np.stack(psds_z[:-1]))
        psds.append(np.stack(psds_a[:-1]))
        fpeaks.append(np.stack(fpeaks_x[:-1]))
        fpeaks.append(np.stack(fpeaks_y[:-1]))
        fpeaks.append(np.stack(fpeaks_z[:-1]))
        fpeaks.append(np.stack(fpeaks_a[:-1]))
    return psds, fpeaks, wlabels, wfeatures, freqs

def load_subject(subject_id,ids_file,folder):
    ids_file = pd.read_csv(folder+ids_file)
    subject_psds = list()
    subject_fpeaks = list()
    subject_wlabels = list()
    subject_wfeatures = list()
    measurements_labels_medication = list()
    measurements_labels_dyskinesia = list()
    measurements_labels_tremor = list()
    measurements_labels = dict()
    twfeatures = list()
    n = ids_file[ids_file['subject_id'] == subject_id].values[:,0].shape[0]
    i = 0
    p1 = 0
    for measurement_id in ids_file[ids_file['subject_id'] == subject_id].values[:,0]:
        psds, fpeaks, wlabels, wfeatures, freqs = load_spectrums(measurement_id,folder,all_psds=False)        
        wfeatures[np.isnan(wfeatures)] = 0
        subject_psds.append(psds[0])
        subject_fpeaks.append(fpeaks[0])
        subject_wlabels.append(wlabels)
        subject_wfeatures.append(wfeatures)
        twfeatures.append(wfeatures[np.where(wlabels==1),:][0,:,:])
        measurements_labels_medication.append(ids_file['on_off'][ids_file['measurement_id'] == measurement_id].values.astype(int))
        measurements_labels_dyskinesia.append(ids_file['dyskinesia'][ids_file['measurement_id'] == measurement_id].values.astype(int))
        measurements_labels_tremor.append(ids_file['tremor'][ids_file['measurement_id'] == measurement_id].values.astype(int))
        i+=1
        p2 = int((1-(n-i)/n)*100)
        if p2>p1:
            print("{}: {}".format(subject_id,p2)+"%")
            p1 = p2
    twfeatures = np.vstack(twfeatures)
    measurements_labels_medication = np.stack(measurements_labels_medication)
    measurements_labels_dyskinesia = np.stack(measurements_labels_dyskinesia)
    measurements_labels_tremor = np.stack(measurements_labels_tremor)
    measurements_labels = np.hstack((measurements_labels_medication,measurements_labels_dyskinesia))
    measurements_labels = np.hstack((measurements_labels,measurements_labels_tremor))
    
    return subject_psds,subject_fpeaks,subject_wlabels,subject_wfeatures,measurements_labels,freqs,twfeatures[1:,:]

#-----------------------------------------------------------------------------
#%%
    
if __name__ == '__main__':

    subjects_ids = [1006]#[1032,1049]#[1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051]
    ids_file = "CIS-PD_Training_Data_IDs_Labels.csv"
    folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
    f = hdf5_handler(folder+'training_data_preclustering_subset.hdf5','a')

    for subject_id in subjects_ids:
        print('Loading subject '+str(subject_id))
        subj = f.create_group(str(subject_id))
        measurements = subj.create_group('measurements')
        data = load_subject(subject_id,ids_file,folder)
        for i in range(0,len(data[0])):
            if i < 100:
                if i < 10:
                    n = '00'+str(i)
                else:
                    n = '0'+str(i)
            else:
                n = str(i)
            measurements.create_dataset('PSD'+n,data=data[0][i])
            measurements.create_dataset('fpeaks'+n, data=data[1][i])
            measurements.create_dataset('wlabels'+n,data=data[2][i])
            measurements.create_dataset('wfeatures'+n, data=data[3][i])
        subj.create_dataset('labels', data=data[4])
        subj.create_dataset('freqs', data=data[5])
        subj.create_dataset('twfeatures', data=data[6])
    
    print('Prepare data done!')
