import numpy as np
import h5py
import contextlib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy.random import randint, choice
from itertools import compress


def threshold_data(data,labels,threshold=100):
    valid_data = list()
    valid_labels = list()
    for i in range(0,len(data)):
        if (data[i].shape[0] >= threshold):
            valid_data.append(data[i][:threshold,:])
            valid_labels.append(labels[i])
    return valid_data, valid_labels


def get_batch(data, labels):
    valid_data, valid_labels = threshold_data(data,labels)
    X_train, X_test, y_train, y_test = train_test_split(valid_data, valid_labels, test_size=0.25)
    # X_test = np.stack(X_test)
    # y_test = np.stack(y_test)
    pairs, targets = get_pairs(X_train,y_train)
   
    return pairs, targets, X_train, X_test, y_train, y_test


def get_pairs(data,labels):
    id_labels = [0,1,2,3,4]
    l_matched = list()
    r_matched = list()
    l_unmatched = list()
    r_unmatched = list()
    m_labels = list()
    u_labels = list()
    cat_data = dict()
    n = int(len(data)/4)
    
    for i in range(0,n):
        j = randint(0,len(data))
        l_matched.append(data.pop(j))
        m_labels.append(labels.pop(j))
        j = randint(0,len(data))
        l_unmatched.append(data.pop(j))
        u_labels.append(labels.pop(j))
    
    for i in range(0,5):
        cat_data[i] = list(compress(data,np.asarray(labels) == i))
    
    l_m_pop = np.ones(len(l_matched))
    for i in range(0,len(l_matched)):
        j = m_labels[i]#[0]
        if len(cat_data[j])>0:
            m = cat_data[j].pop(0)
            r_matched.append(m)
        else:
            l_m_pop[i] = 0  
    l_matched = list(compress(l_matched,l_m_pop.astype(bool)))        
    
    l_u_pop = np.ones(len(l_unmatched))
    for i in range(0,len(l_unmatched)):
        c_labels = list(compress(id_labels,np.asarray(id_labels) != u_labels[i]))#[0]))
        if len(cat_data[c_labels[0]])>0:
            u = cat_data[c_labels[0]].pop(0)
            r_unmatched.append(u)
        elif len(cat_data[c_labels[1]])>0:
            u = cat_data[c_labels[1]].pop(0)
            r_unmatched.append(u)
        elif len(cat_data[c_labels[2]])>0:
            u = cat_data[c_labels[2]].pop(0)
            r_unmatched.append(u)
        elif len(cat_data[c_labels[3]])>0:
            u = cat_data[c_labels[3]].pop(0)
            r_unmatched.append(u)
        else:
            l_u_pop[i] = 0
    l_unmatched = list(compress(l_unmatched,l_u_pop.astype(bool)))
    
    targets = np.hstack((np.ones(len(l_matched)),np.zeros(len(l_unmatched))))
    l_matched = np.stack(l_matched)
    l_unmatched = np.stack(l_unmatched)
    l_pairs = np.vstack((l_matched,l_unmatched))
    r_matched = np.stack(r_matched)
    r_unmatched = np.stack(r_unmatched)
    r_pairs = np.vstack((r_matched,r_unmatched))
    l_pairs = l_pairs.reshape(l_pairs.shape[0],l_pairs.shape[1],l_pairs.shape[2],1)
    r_pairs = r_pairs.reshape(r_pairs.shape[0],r_pairs.shape[1],r_pairs.shape[2],1)
    pairs = [l_pairs,r_pairs]
    
    return pairs, targets


def make_oneshot_task(val_data,val_labels,N):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    _, w, h = val_data.shape
    
    test_label = randint(0,4)
    
    # Matched and unmatched candidates
    i_m = (np.asarray(val_labels)==test_label)[:,0]
    i_u = (np.asarray(val_labels)!=test_label)[:,0]
    m_candidates = val_data[i_m,:,:]
    u_candidates = val_data[i_u,:,:]
    
    # Random indices for sampling
    try:
        m_idx1, m_idx2 = choice(m_candidates.shape[0],replace=False,size=(2,)) # Non repetitive random indices
        u_indices = randint(0,u_candidates.shape[0],size=(N,))
    except:
        print('Not enough validation candidates')

    # Matched image from support_set will be allocated to position '0' then shuffled
    test_image = np.asarray([m_candidates[m_idx1,:,:]]*N).reshape(N, w, h, 1)
    support_set = u_candidates[u_indices,:,:]
    support_set[0,:,:] = m_candidates[m_idx2,:,:]
    support_set = support_set.reshape(N, w, h, 1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)

    pairs = [test_image, support_set]

    return pairs, targets

  
def test_oneshot(model,val_data,val_labels,N,k, verbose = 0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for _ in range(k):
        # try:
        inputs, targets = make_oneshot_task(val_data,val_labels,N)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
        # except:
        #     print('Not enough validation candidates')
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
    return percent_correct


def test_model(model,val_data,val_labels):

    val_data = val_data.reshape(val_data.shape[0],1,val_data.shape[1],val_data.shape[2],1)
    n = val_labels.shape[0]
    n_correct = 0
    k = 0
    for i in range(0,n):
        k = 0
        for j in range(1+k,n):
            truth = int(val_labels[i]==val_labels[j])
            inputs = (val_data[i,:,:,:,:],val_data[j,:,:,:,:])
            pred = int(np.around(model.predict(inputs)))
            if (truth==pred):
                n_correct += 1
            k += 1
    accuracy = (n_correct/((pow(n,2)-n)/2))*100
    tps = list()
    tps.append(np.sum((val_labels==0).astype(int)))
    tps.append(np.sum((val_labels==1).astype(int)))
    tps.append(np.sum((val_labels==2).astype(int)))
    tps.append(np.sum((val_labels==3).astype(int)))
    tps.append(np.sum((val_labels==4).astype(int)))

    return accuracy, n, tps
    

def test_model_2(model, train_data, val_data, train_labels, val_labels):

    train_data_0 = np.stack(list(compress(train_data,np.asarray(train_labels)==0)))
    train_data_0 = train_data_0.reshape(train_data_0.shape[0],train_data_0.shape[1],train_data_0.shape[2],1)
    train_data_1 = np.stack(list(compress(train_data,np.asarray(train_labels)==1)))
    train_data_1 = train_data_1.reshape(train_data_1.shape[0],train_data_1.shape[1],train_data_1.shape[2],1)
    train_data_2 = np.stack(list(compress(train_data,np.asarray(train_labels)==2)))
    train_data_2 = train_data_2.reshape(train_data_2.shape[0],train_data_2.shape[1],train_data_2.shape[2],1)
    train_data_3 = np.stack(list(compress(train_data,np.asarray(train_labels)==3)))
    train_data_3 = train_data_3.reshape(train_data_3.shape[0],train_data_3.shape[1],train_data_3.shape[2],1)
    train_data_4 = np.stack(list(compress(train_data,np.asarray(train_labels)==4)))
    train_data_4 = train_data_4.reshape(train_data_4.shape[0],train_data_4.shape[1],train_data_4.shape[2],1)

    n_labels = [train_data_0.shape[0],train_data_1.shape[0],train_data_2.shape[0],train_data_3.shape[0],train_data_4.shape[0]]
    
    val_data = np.stack(val_data)

    n_correct = 0

    for i in range(0,val_data.shape[0]):
        val_data_0 = ((np.asarray([val_data[i,:,:]]*train_data_0.shape[0]).reshape(train_data_0.shape[0],val_data.shape[1],val_data.shape[2],1)))
        val_data_0 = [val_data_0,train_data_0]
        val_data_1 = ((np.asarray([val_data[i,:,:]]*train_data_1.shape[0]).reshape(train_data_1.shape[0],val_data.shape[1],val_data.shape[2],1)))
        val_data_1 = [val_data_1,train_data_1]
        val_data_2 = ((np.asarray([val_data[i,:,:]]*train_data_2.shape[0]).reshape(train_data_2.shape[0],val_data.shape[1],val_data.shape[2],1)))
        val_data_2 = [val_data_2,train_data_2]
        val_data_3 = ((np.asarray([val_data[i,:,:]]*train_data_3.shape[0]).reshape(train_data_3.shape[0],val_data.shape[1],val_data.shape[2],1)))
        val_data_3 = [val_data_3,train_data_3]
        val_data_4 = ((np.asarray([val_data[i,:,:]]*train_data_4.shape[0]).reshape(train_data_4.shape[0],val_data.shape[1],val_data.shape[2],1)))
        val_data_4 = [val_data_4,train_data_4]

        inputs = [val_data_0,val_data_1,val_data_2,val_data_3,val_data_4]

        preds = list()
        
        for j in range(0,len(inputs)):
            pred = np.sum(np.around(model.predict(inputs[j])))
            if (n_labels[j] > 0):
                preds.append(pred/n_labels[j])
            else:
                preds.append(-1)

        if np.argmax(np.stack(preds))==val_labels[i]:
            n_correct += 1

    
    accuracy = n_correct/len(val_labels)

    return accuracy
    
    
def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename.encode(), fapl=propfaid)) as fid:
        return h5py.File(fid, mode)
