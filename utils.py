import numpy as np
import h5py
import contextlib

def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename.encode(), fapl=propfaid)) as fid:
        return h5py.File(fid, mode)

def load_twfeatures(path,subject):
    file = path+"training_data_preclustering_1004.hdf5"
    f = hdf5_handler(file,'r')
    twfeatures = f[str(subject)]['twfeatures'][()]
    f.close()
    return twfeatures

def load_pf_hists(path,subject,classifier,scaler=False,pca=False,ft0=0,ft1=100):
    file = path+"training_data_preclustering_1004.hdf5"
    f = hdf5_handler(file,'r')
    data = f[str(subject)]['measurements']
    labels = f[str(subject)]['labels'][()]
    freqs = f[str(subject)]['freqs'][()]
    k = [key for key in data.keys()]
    n = int(len(k)/6)
    n_clusters = classifier.n_clusters
    # psds_list = k[0:n]
    disps_list = k[n:2*n]
    fpeaks_list = k[2*n:3*n]
    pratios_list = k[3*n:4*n]
    wfeatures_list = k[4*n:5*n]
    wlabels_list = k[5*n:6*n]
    pf_hists = list()
    for i in range(0,n):
        wfeatures = data[wfeatures_list[i]][()]
        disps = data[disps_list[i]][()]
        fpeaks = data[fpeaks_list[i]][()]
        pratios = data[pratios_list[i]][()]
        wlabels = data[wlabels_list[i]][()]
        rntw = np.sum(wlabels)/wlabels.shape[0]
        twfeatures = wfeatures[np.where(wlabels==1)][:,ft0:ft1+1]
        if scaler:
            twfeatures = scaler.transform(twfeatures)
        if pca:
            twfeatures = pca.transform(twfeatures)
        twclusts = classifier.predict(twfeatures)
        tdisps = disps[np.where(wlabels==1)]
        tfpeaks = fpeaks[np.where(wlabels==1)]
        hist = list()
        for i in range(0,n_clusters):
            n_m = np.sum(twclusts==i)
            if n_m > 0:
                tdisps_i = tdisps[np.where(twclusts==i)]#[:,4]
                tdisps_i = np.sum(tdisps_i,axis=0)/n_m
            else:
                tdisps_i = np.zeros(8)
            tfpeaks_i = tfpeaks[np.where(twclusts==i)]
            hist_tfpeaks = np.zeros(freqs.shape[0])
            dict_tfpeaks = {i:0 for i in np.unique(tfpeaks_i)}
            for i in range(0,tfpeaks_i.shape[0]):
                dict_tfpeaks[tfpeaks_i[i]] += 1/tfpeaks_i.shape[0]
            for i in dict_tfpeaks.keys():
                hist_tfpeaks[freqs==i] = dict_tfpeaks[i]
            hist.append(hist_tfpeaks)
            hist.append(tdisps_i)
        hist.append(rntw)
        pf_hists.append(np.hstack(hist))
    pf_hists = np.stack(pf_hists)
    
    return pf_hists,labels

    
    
    
    
    
    
    
    
    
    
    