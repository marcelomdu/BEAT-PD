import numpy as np
import scipy.sparse as sp
import torch
import h5py
import contextlib
from numpy.random import randint

from scipy.sparse.linalg.eigen.arpack import eigsh


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

def get_balanced_indexes(labels,n_val=2):

    idx_train = list()
    idx_val = list()
    idx_test = list()

    id_labels = np.unique(labels)
    idx_labels = {i:[] for i in id_labels}
    n_labels = {i:[] for i in id_labels}
    for i in id_labels:
        idx_labels[i] = np.where(labels==i)[0].tolist()
        n_labels[i] = len(idx_labels[i])

    for j in id_labels:
        if n_labels[j] > n_val*4:
            for i in range(0,n_val):
                idx_val.append(idx_labels[j].pop(randint(0,len(idx_labels[j]))))
                idx_test.append(idx_labels[j].pop(randint(0,len(idx_labels[j]))))
        idx_train = np.hstack((idx_train,idx_labels[j])).astype(np.uint8)

    idx_train = torch.LongTensor(idx_train.tolist())
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return idx_train, idx_val, idx_test

def load_data(path="/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/", 
            study="CIS", 
            subfolder=None, 
            subject="1004", 
            label="tre", 
            cn_type="1",
            threshold=20):

    if subfolder is not None:
        file = path+study+"/Train/training_data/"+subfolder+"/training_data_graphs.hdf5"
    else:
        file = path+study+"/Train/training_data/"+"training_data_graphs.hdf5"
    f = hdf5_handler(file,'r')

    if label == "dys":
        n_alvo = '1'
        n_ft1 = '2'
        n_ft2 = '3'
    if label == "med":
        n_alvo = '2'
        n_ft1 = '3'
        n_ft2 = '1'
    if label == "tre":
        n_alvo = '3'
        n_ft1 = '1'
        n_ft2 = '2'

    data = f[str(subject)]
    labels = data['ft_matrix{}'.format(n_alvo)][()]
    ft_1 = data['ft_matrix{}'.format(n_ft1)][()]
    ft_2 = data['ft_matrix{}'.format(n_ft2)][()]
    if (labels.shape[1] > 0):
        y = torch.LongTensor(np.where(labels)[1])
        if (ft_1.shape[1] > 0) & (ft_2.shape[1] > 0):
            x = torch.FloatTensor(np.hstack((ft_1,ft_2))) # nodes' features matrix N x F
        elif (ft_1.shape[1] > 0):
            x = torch.FloatTensor(ft_1)
        elif (ft_2.shape[1] > 0):
            x = torch.FloatTensor(ft_2)
        else:
            x = torch.FloatTensor(labels)
    else:
        raise Exception("Invalid target label")
    
    adj, _ = get_adjacency(data['cn_matrix{}'.format(cn_type)][()], 100-threshold)
    idx = torch.from_numpy(np.vstack(np.nonzero(adj)))
    n_adj = torch.from_numpy(np.ones(idx.shape[1]).astype(np.double)).to(dtype=torch.float32)
    adj = torch.sparse.FloatTensor(idx,n_adj,torch.Size([adj.shape[0],adj.shape[1]]))

    idx_train, idx_val, idx_test = get_balanced_indexes(data['labels'][:,int(n_alvo)-1])
    
    return adj, x, y, idx_train, idx_val, idx_test

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
    return correct / len(labels)

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

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

