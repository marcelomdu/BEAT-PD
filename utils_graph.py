import numpy as np
import scipy.sparse as sp
import torch
import h5py
import contextlib
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scipy.sparse.linalg.eigen.arpack import eigsh

from scipy.stats import zscore
from matplotlib import pyplot as plt


def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename.encode(), fapl=propfaid)) as fid:
        return h5py.File(fid, mode)

def get_adjacency(cn_matrix, threshold):
    mask = (cn_matrix > np.percentile(cn_matrix, threshold)).astype(np.uint8)
    nodes, neighbors = np.nonzero(mask)
    sparse_mask = {}
    for i, node in enumerate(nodes):
        if neighbors[i] > node:
            if not node in sparse_mask: 
                sparse_mask[node] = [neighbors[i]]
            else:
                sparse_mask[node].append(neighbors[i])
    return mask, sparse_mask

def get_balanced_indexes(labels,n_val=20,test_data_included=True):

    idx_train = list()
    # idx_val = list()
    idx_test = list()

    id_labels = np.unique(labels)
    if test_data_included:
        id_labels = id_labels[1:]
    idx_labels = {i:[] for i in id_labels}
    n_labels = {i:[] for i in id_labels}
    for i in id_labels:
        idx_labels[i] = np.where(labels==i)[0].tolist()
        n_labels[i] = len(idx_labels[i])

    for j in id_labels:
        if n_labels[j] > n_val*2:
            for i in range(0,n_val):
                # idx_val.append(idx_labels[j].pop(randint(0,len(idx_labels[j]))))
                idx_test.append(idx_labels[j].pop(randint(0,len(idx_labels[j]))))
        idx_train = np.hstack((idx_train,idx_labels[j])).astype(np.uint8)

    idx_train = torch.LongTensor(idx_train.tolist())
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return idx_train, idx_test

def load_data(path, subject, label, cn_type, ft_type):

    file = path+"train_test_data_graphs_2.hdf5"

    f = hdf5_handler(file,'r')

    if label == "dys":
        n_alvo = 1
    if label == "med":
        n_alvo = 2
    if label == "tre":
        n_alvo = 3

    data = f[str(subject)]

    scaler = StandardScaler()
    features = data['ft_matrix{}'.format(ft_type)][()]
    features = scaler.fit_transform(features)

    enc = OneHotEncoder(sparse=False)
    labels = data['labels'][()][:,n_alvo-1].reshape(-1,1)
    labels = enc.fit_transform(labels)

    if (labels.shape[1] > 1):
        y = torch.LongTensor(np.where(labels)[1])
        x = torch.FloatTensor(features)
    else:
        raise Exception("Invalid target label")
    
    # adj, _ = get_adjacency(data['cn_matrix{}'.format(cn_type)][()], 100-threshold)
    # idx = torch.from_numpy(np.vstack(np.nonzero(adj)))
    # n_adj = torch.from_numpy(np.ones(idx.shape[1]).astype(np.double)).to(dtype=torch.float32)
    # adj = torch.sparse.FloatTensor(idx,n_adj,torch.Size([adj.shape[0],adj.shape[1]]))

    adj = data['cn_matrix{}'.format(cn_type)][()]
    
    
    ####-------------------------cn metrics test-----------------------####
    # # mask = np.invert(np.diag(np.ones(adj.shape[0])).astype(bool))
    # # adj = mask*adj
    # adj2 = scaler.fit_transform(adj)
    # # adj = adj+np.ones(adj.shape[0])
    
    # sorted_labels1 = list()
    # sorted_labels2 = list()
    # sorted_labels3 = list()
    # sorted_labels4 = list()
    # sorted_labels = list()
    # nt = int(np.sum(labels[:,0]))
    # adj_temp = adj[:-nt,:-nt]
    # adj_temp2 = adj2[:-nt,:-nt]
    # for i in range(0,adj_temp.shape[0]):
    #     n = labels[i,:]==1
    #     adj_sorted = np.argsort(adj_temp[i,:])[::-1]
    #     sorted_labels1.append(labels[adj_sorted,n])
    #     # adj_sorted = np.argsort(adj_temp[:,i])[::-1]
    #     # sorted_labels2.append(labels[adj_sorted,n])
    #     adj_sorted = np.argsort(adj_temp2[i,:])[::-1]
    #     sorted_labels3.append(labels[adj_sorted,n])
    #     # adj_sorted = np.argsort(adj_temp2[:,i])[::-1]
    #     # sorted_labels4.append(labels[adj_sorted,n])

    # sorted_labels.append(np.cumsum(zscore(np.gradient(np.cumsum(np.sum(np.stack(sorted_labels1),axis=0))))))
    # # sorted_labels.append(np.cumsum(zscore(np.gradient(np.cumsum(np.sum(np.stack(sorted_labels2),axis=0))))))
    # sorted_labels.append(np.cumsum(zscore(np.gradient(np.cumsum(np.sum(np.stack(sorted_labels3),axis=0))))))
    # # sorted_labels.append(np.cumsum(zscore(np.gradient(np.cumsum(np.sum(np.stack(sorted_labels4),axis=0))))))

    # for i in range(0,len(sorted_labels)):    
    #     plt.figure(i)
    #     plt.plot(sorted_labels[i])

    
    adj = sp.coo_matrix(adj)
    idx = torch.from_numpy(np.stack([adj.row.astype(np.int_),adj.col.astype(np.int_)]))
    values = torch.from_numpy(adj.data)
    adj = torch.sparse.FloatTensor(idx,values,torch.Size(adj.shape))

    idx_train, idx_test = get_balanced_indexes(data['labels'][:,int(n_alvo)-1])
    
    return adj, x, y, idx_train, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct/len(labels)

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for _ in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def threshold_adj(adj,threshold):  
    adj_temp = adj.to_dense().numpy()
    adj_temp, _ = get_adjacency(adj_temp, 100-threshold)
    adj_temp = sp.coo_matrix(adj_temp.astype(np.double))
    idx = torch.from_numpy(np.stack([adj_temp.row.astype(np.int_),adj_temp.col.astype(np.int_)]))
    values = torch.from_numpy(adj_temp.data)
    adj = torch.sparse.FloatTensor(idx,values,torch.Size(adj_temp.shape))

    return adj