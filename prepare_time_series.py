#%%
import contextlib
import h5py
import pandas as pd
import numpy as np
from scipy import signal
from numba import jit

def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename.encode(), fapl=propfaid)) as fid:
        return h5py.File(fid, mode)

@jit
def calc_var_vect_rts(xyz):
    rho = np.zeros(xyz.shape[0]-2)
    thetax = np.zeros(xyz.shape[0]-2)
    thetay = np.zeros(xyz.shape[0]-2)
    thetaz = np.zeros(xyz.shape[0]-2)
    for i in range(1,xyz.shape[0]-1):
        diff = xyz[i,:]-xyz[i-1,:]
        rho[i-1] = np.sqrt(np.power(diff[0],2)+np.power(diff[1],2)+np.power(diff[2],2))
        thetax[i-1] = np.arctan2(diff[2],diff[1])
        thetay[i-1] = np.arctan2(diff[2],diff[0])
        thetaz[i-1] = np.arctan2(diff[1],diff[0])
    rho = np.concatenate((rho,np.zeros(2)))
    thetax = np.concatenate((thetax,np.zeros(2)))
    thetay = np.concatenate((thetay,np.zeros(2)))
    thetaz = np.concatenate((thetaz,np.zeros(2)))
    return rho,thetax,thetay,thetaz

@jit
def mean_downsample(x,wlen):
    x2 = np.zeros((1,x.shape[1]))
    for i in range(0,x.shape[0],int(wlen)):
        xtemp = x[i:i+wlen,:]
        xmean = np.mean(xtemp,axis=0)
        x2 = np.vstack((x2,xmean))
    return x2[1:,:]

def load_time_series(x,folder):
    subj = pd.read_csv(folder+x+".csv")
    if 'x' in subj.columns:
        subj = subj[['t','x','y','z']]
    elif 'X' in subj.columns:
        subj = subj[['Timestamp','X','Y','Z']]
    # Calculate A as the magnitude of (x,y,z) vector
    x2y2 = np.add(np.power(subj.values[:,1],2),np.power(subj.values[:,2],2))
    subj['A'] = np.sqrt(np.add(x2y2,np.power(subj.values[:,3],2))) # Rho
    subj['Tx'] = np.arctan2(subj.values[:,3],subj.values[:,2])  # Rotation around x
    subj['Ty'] = np.arctan2(subj.values[:,3],subj.values[:,1])  # Rotation around y
    subj['Tz'] = np.arctan2(subj.values[:,2],subj.values[:,1]) # Rotation around z
    subj['Rho'],subj['Thetax'],subj['Thetay'],subj['Thetaz'] = calc_var_vect_rts(subj[['X','Y','Z']].values)
    # Median filter
    subj['X_fm'] = signal.medfilt(subj['X'].values,kernel_size=[5])
    subj['Y_fm'] = signal.medfilt(subj['Y'].values,kernel_size=[5])
    subj['Z_fm'] = signal.medfilt(subj['Z'].values,kernel_size=[5])
    subj['A_fm'] = signal.medfilt(subj['A'].values,kernel_size=[5])
    subj['Tx_fm'] = signal.medfilt(subj['Tx'].values,kernel_size=[5])
    subj['Ty_fm'] = signal.medfilt(subj['Ty'].values,kernel_size=[5])
    subj['Tz_fm'] = signal.medfilt(subj['Tz'].values,kernel_size=[5])
    subj['Rho_fm'],subj['Thetax_fm'],subj['Thetay_fm'],subj['Thetaz_fm'] = calc_var_vect_rts(subj[['X_fm','Y_fm','Z_fm']].values)
    # Bandpass Butterworth filter
    sosb1 = signal.butter(5,[1,20],btype='bandpass',fs=50,output='sos')
    subj['X_fb'] = signal.sosfilt(sosb1,subj['X_fm'].values)
    subj['Y_fb'] = signal.sosfilt(sosb1,subj['Y_fm'].values)
    subj['Z_fb'] = signal.sosfilt(sosb1,subj['Z_fm'].values)
    subj['A_fb'] = signal.sosfilt(sosb1,subj['A_fm'].values)
    subj['Tx_fb'] = signal.sosfilt(sosb1,subj['Tx_fm'].values)
    subj['Ty_fb'] = signal.sosfilt(sosb1,subj['Ty_fm'].values)
    subj['Tz_fb'] = signal.sosfilt(sosb1,subj['Tz_fm'].values)
    subj['Rho_fb'],subj['Thetax_fb'],subj['Thetay_fb'],subj['Thetaz_fb'] = calc_var_vect_rts(subj[['X_fb','Y_fb','Z_fb']].values)
    # PCA for main motion axis extraction (not included in Ortega et al.)
    time_series = mean_downsample(subj.values[:,1:],wlen=50)
    
    return time_series

def load_subject(subject_id,ids_file,folder):
    ids_file = pd.read_csv(folder+ids_file)
    subject_time_series = list()
    measurements_labels_medication = list()
    measurements_labels_dyskinesia = list()
    measurements_labels_tremor = list()
    measurements_labels = dict()
    n = ids_file[ids_file['subject_id'] == subject_id].values[:,0].shape[0]
    i = 0
    p1 = 0
    for measurement_id in ids_file[ids_file['subject_id'] == subject_id].values[:,0]:
        time_series = load_time_series(measurement_id,folder)        
        subject_time_series.append(time_series)
        measurements_labels_medication.append(ids_file['on_off'][ids_file['measurement_id'] == measurement_id].values.astype(int))
        measurements_labels_dyskinesia.append(ids_file['dyskinesia'][ids_file['measurement_id'] == measurement_id].values.astype(int))
        measurements_labels_tremor.append(ids_file['tremor'][ids_file['measurement_id'] == measurement_id].values.astype(int))
        i+=1
        p2 = int((1-(n-i)/n)*100)
        if p2>p1:
            print("{}: {}".format(subject_id,p2)+"%")
            p1 = p2
    measurements_labels_medication = np.stack(measurements_labels_medication)
    measurements_labels_dyskinesia = np.stack(measurements_labels_dyskinesia)
    measurements_labels_tremor = np.stack(measurements_labels_tremor)
    measurements_labels = np.hstack((measurements_labels_medication,measurements_labels_dyskinesia))
    measurements_labels = np.hstack((measurements_labels,measurements_labels_tremor))
    
    return subject_time_series,measurements_labels

#-----------------------------------------------------------------------------
#%%
    
if __name__ == '__main__':

    subjects_ids = [1004]#[1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051]
    ids_file = "CIS-PD_Training_Data_IDs_Labels.csv"
    folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
    f = hdf5_handler(folder+'training_data_lstm.hdf5','a')

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
            measurements.create_dataset('time_series'+n,data=data[0][i])
        subj.create_dataset('labels', data=data[1])
    
    print('Prepare data done!')
