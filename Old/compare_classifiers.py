# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
#%%
import contextlib
import h5py
import pandas as pd
from scipy import signal
from numba import njit
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import plot_confusion_matrix

#%%
#--------------------------------------------------------------------------------------------------------------------------------------
# Prepare data
#--------------------------------------------------------------------------------------------------------------------------------------
def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename.encode(), fapl=propfaid)) as fid:
        return h5py.File(fid, mode)

def calc_mag_diff(x):
    mag_diff = np.zeros(x.shape[0]-2)
    for i in range(1,x.shape[0]-1):
        diff = x[i,:]-x[i-1,:]
        mag_diff[i-1] = np.sqrt(np.power(diff[0],2)+np.power(diff[1],2)+np.power(diff[2],2))
    mag_diff = np.concatenate((mag_diff,np.zeros(2)))
    
    return mag_diff

def load_spectrums(x,folder,interval=4,lf=4,hf=8,th=0.6):
    samples = interval*50 # interval in seconds
    subj = pd.read_csv(folder+x+".csv",usecols=(1,2,3))
    sos0 = signal.butter(10,[2,20],btype='bandpass',fs=50,output='sos')
    subj['x_filt'] = signal.sosfilt(sos0,subj['X'])
    subj['y_filt'] = signal.sosfilt(sos0,subj['Y'])
    subj['z_filt'] = signal.sosfilt(sos0,subj['Z'])
    
    sos = signal.butter(10,[4,8],btype='bandpass',fs=50,output='sos')
    # PCA
    pca = PCA(n_components=1)
    subj_val = subj.values[:,3:6]
    subj['pca_axis'] = pca.fit_transform(subj_val)
    subj['filtered_pca_axis'] = signal.sosfilt(sos,subj['pca_axis'].values)
    # Mag diff
    subj['mag_diff'] = calc_mag_diff(subj_val)
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
    tau = nperseg/5
    window = signal.windows.exponential(nperseg,tau=tau)
    # window = 'hann'
    for i in range(0,subj.values.shape[0],int(samples/2)):
        sig1 = subj['filtered_pca_axis'].values[i:i+samples]
        sig2 = subj['filtered_mag_diff'].values[i:i+samples]
        if sig1.shape[0] == samples:
            # PCA PSDs
            freqs, psd_pca = signal.welch(sig1,fs=50,window=window,detrend='linear',nperseg=nperseg)
            argmax_pca = np.argmax(psd_pca[lf:hf])+lf
            fpeaks_pca.append(np.around(freqs[argmax_pca],decimals=2))
            pratio_pca = np.sum(psd_pca[argmax_pca-1:argmax_pca+1])/np.sum(psd_pca[lf:hf])
            if pratio_pca > th:
                wlabels_pca.append(1)
            else:
                wlabels_pca.append(0)
            psd_pca = psd_pca[lf-3:hf+3]#/np.max(psd_pca[lf-3:hf+3])
            psds_pca.append(psd_pca)
            # Mag diff PSDs
            freqs, psd_mag = signal.welch(sig2,fs=50,window=window,detrend='linear',nperseg=nperseg)
            argmax_mag = np.argmax(psd_mag[lf:hf])+lf
            fpeaks_mag.append(np.around(freqs[argmax_mag],decimals=2))
            pratio_mag = np.sum(psd_mag[argmax_mag-1:argmax_mag+1])/np.sum(psd_mag[lf:hf])
            if pratio_mag > th:
                wlabels_mag.append(1)
            else:
                wlabels_mag.append(0)
            psd_mag = psd_mag[lf-3:hf+3]#/np.max(ps_mag[lf-3:hf+3])
            psds_mag.append(psd_mag)
    freqs = freqs[lf:hf]
    wlabels_pca = np.stack(wlabels_pca[:-1])
    fpeaks_pca = np.stack(fpeaks_pca[:-1])
    psds_pca = np.stack(psds_pca[:-1])
    wlabels_mag = np.stack(wlabels_mag[:-1])
    fpeaks_mag = np.stack(fpeaks_mag[:-1])
    psds_mag = np.stack(psds_mag[:-1])
    # no repeated peaks
    # tfpeaks_and = fpeaks_mag[np.where(np.logical_and(wlabels_pca==1,wlabels_mag==1))]
    # tfpeaks_pca = fpeaks_pca[np.where(np.logical_and(wlabels_pca==1,np.logical_xor(wlabels_pca==1,wlabels_mag==1)))]
    # tfpeaks_mag = fpeaks_mag[np.where(np.logical_and(wlabels_mag==1,np.logical_xor(wlabels_pca==1,wlabels_mag==1)))]
    # tfpeaks = np.hstack((tfpeaks_and,np.hstack((tfpeaks_pca,tfpeaks_mag))))
    tfpeaks_pca = fpeaks_pca[np.where(wlabels_pca==1)]
    # tfpeaks_mag = fpeaks_mag[np.where(wlabels_mag==1)]
    tfpeaks = tfpeaks_pca #np.hstack((tfpeaks_pca,tfpeaks_mag))
    
    tufpeaks = np.unique(tfpeaks)
    dict_tfpeaks = {i:0 for i in tufpeaks}
    for i in range(0,tfpeaks.shape[0]):
        dict_tfpeaks[tfpeaks[i]] += 1/tfpeaks.shape[0]
    hist_tfpeaks = np.zeros(freqs.shape[0])
    for i in dict_tfpeaks.keys():
        hist_tfpeaks[freqs==i] = dict_tfpeaks[i]
    nw = fpeaks_pca.shape[0]
    tw = tfpeaks.shape[0]
    rntw = tw/nw

    return hist_tfpeaks,freqs,rntw,tw,nw

def load_subject(subject_id,ids_file,folder):
    ids_file = pd.read_csv(folder+ids_file)
    subject_histfs = list()
    subject_rntws = list()
    subject_tw = list()
    subject_nw = list()
    measurements_labels_medication = list()
    measurements_labels_dyskinesia = list()
    measurements_labels_tremor = list()
    measurements_labels = dict()
    n = ids_file[ids_file['subject_id'] == subject_id].values[:,0].shape[0]
    i = 0
    per1 = -1
    for measurement_id in ids_file[ids_file['subject_id'] == subject_id].values[:,0]:
        hist_tfpeaks,freqs,rntw,tw,nw = load_spectrums(measurement_id,folder)
        subject_histfs.append(hist_tfpeaks)
        subject_rntws.append(rntw)
        subject_tw.append(tw)
        subject_nw.append(nw)
        measurements_labels_medication.append(ids_file['on_off'][ids_file['measurement_id'] == measurement_id].values.astype(int))
        measurements_labels_dyskinesia.append(ids_file['dyskinesia'][ids_file['measurement_id'] == measurement_id].values.astype(int))
        measurements_labels_tremor.append(ids_file['tremor'][ids_file['measurement_id'] == measurement_id].values.astype(int))
        i+=1
        per2 = int((1-(n-i)/n)*100)
        if per2 > per1: 
            print("{}".format(per2)+"%")
            per1 = per2
    measurements_labels_medication = np.stack(measurements_labels_medication)
    measurements_labels_dyskinesia = np.stack(measurements_labels_dyskinesia)
    measurements_labels_tremor = np.stack(measurements_labels_tremor)
    measurements_labels = np.hstack((measurements_labels_medication,measurements_labels_dyskinesia))
    measurements_labels = np.hstack((measurements_labels,measurements_labels_tremor))
    subject_histfs = np.stack(subject_histfs)
    subject_rntws = np.stack(subject_rntws)
    
    return subject_histfs, subject_rntws, measurements_labels, subject_tw, subject_nw

#%%
#--------------------------------------------------------------------------------------------------------------------------------------
# Load data
#--------------------------------------------------------------------------------------------------------------------------------------

subjects_ids = [1032]#,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051]
ids_file = "CIS-PD_Training_Data_IDs_Labels.csv"
folder = "/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
for subject_id in subjects_ids:
    print('Loading subject '+str(subject_id))
    data = load_subject(subject_id,ids_file,folder)  

#%%
# Select data
d0 = data[0]
d1 = data[1]
X_data = np.column_stack((d0,d1))
# X_data = StandardScaler().fit_transform(X_data)
# pca = PCA(n_components=2)
# X_data = pca.fit_transform(X_data)

y_data = data[2][:,2]

X = X_data[np.where(np.logical_or(y_data==1,y_data==0))]#[:,1:3]
y = y_data[np.where(np.logical_or(y_data==1,y_data==0))]

datasets = [[X,y]]

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, class_weight='balanced'),
    SVC(gamma=2, C=1, class_weight='balanced'),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    # x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    # y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                      np.arange(y_min, y_max, h))

    # # just plot the dataset first
    # cm = plt.cm.RdBu
    # cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # if ds_cnt == 0:
    #     ax.set_title("Input data")
    # # Plot the training points
    # ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
    #            edgecolors='k')
    # # Plot the testing points
    # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
    #            edgecolors='k')
    # ax.set_xlim(xx.min(), xx.max())
    # ax.set_ylim(yy.min(), yy.max())
    # ax.set_xticks(())
    # ax.set_yticks(())
    # i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):

        # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("{}\t\t{}".format(name,score))
        
        class_names = [0,1]
    
        np.set_printoptions(precision=2)
        
        # Plot non-normalized confusion matrix
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues)
        disp.ax_.set_title(name)
    
        # print(disp.confusion_matrix)
        
        plt.show()

        # # Plot the decision boundary. For that, we will assign a color to each
        # # point in the mesh [x_min, x_max]x[y_min, y_max].
        # if hasattr(clf, "decision_function"):
        #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        # else:
        #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # # Put the result into a color plot
        # Z = Z.reshape(xx.shape)
        # ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # # Plot the training points
        # ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
        #            edgecolors='k')
        # # Plot the testing points
        # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
        #            edgecolors='k', alpha=0.6)

        # ax.set_xlim(xx.min(), xx.max())
        # ax.set_ylim(yy.min(), yy.max())
        # ax.set_xticks(())
        # ax.set_yticks(())
        # if ds_cnt == 0:
        #     ax.set_title(name)
        # ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
        #         size=15, horizontalalignment='right')
        # i += 1

# plt.tight_layout()
# plt.show()
#%%