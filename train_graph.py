from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_graph import load_data, accuracy, chebyshev_polynomials, threshold_adj
from models_graph import GCN, ChebyGCN, GeoChebConv, GeoSAGEConv, GeoGATConv

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--model', default='gcn_cheby', help='gcn model used (default: gcn_cheby, '
                                                             'uses chebyshev polynomials, '
                                                             'options: gcn, gcn_cheby, dense )')
parser.add_argument('--study', type=str, default="CIS",
                    help='Study name')
parser.add_argument('--condition', type=str, default="tre",
                    help='Condition for training (med: medication, dys: dyskinesia, tre: tremor)')
parser.add_argument('--cn_type', type=int, default=5,
                    help='Connectivity to be used (1:PSD cross-correlation, 2:PSD derivative cross-correlation, 3:PSD 2nd derivative cross-correlation)')
parser.add_argument('--ft_type', type=int, default=4,
                    help='Features to be used (1:PSDs, 2:PSDs first derivative, 3:PSDs 2nd derivative, 4:Signal statistical features)')
parser.add_argument('--cn_threshold', type=float, default=20,
                    help='Threshold to be used for the connectivity matrix')
parser.add_argument('--subject', default=None,
                    help='Choose single subject for train')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


seed = args.seed
seed = np.random.randint(100)
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

study = args.study
cn_type = args.cn_type
ft_type = args.ft_type
cn_threshold = args.cn_threshold
label=args.condition

if study == "CIS":
    path="/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
    subjects_list = [1004,1006,1007,1019,1020,1023,1032,1034,1038,1046,1048,1049] #1051,1044,1039,1043

if study == "REAL":
    path="/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/REAL/Train/training_data/smartwatch_accelerometer/"
    subjects_list = ['hbv012', 'hbv013', 'hbv022', 'hbv023', 'hbv038','hbv054']#'hbv017', 'hbv051',  'hbv077', 'hbv043', 'hbv014', 'hbv018', 

if not (args.subject == None):
    subjects_list = [args.subject]

subjects_list = [1038]

for subject in subjects_list:

    # Load data
    adj, features, labels, idx_train, idx_test = load_data(path=path,subject=str(subject),label=label,cn_type=cn_type,ft_type=ft_type)
    
    if cn_threshold > 0:
        adj = threshold_adj(adj,cn_threshold)

    # Model and optimizer
    if args.model == 'gcn':
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)
    elif args.model == 'gcn_cheby':
        adj = adj.to_dense().numpy()
        adj_temp = chebyshev_polynomials(adj, 3)
        adj = torch.Tensor(len(adj_temp),adj.shape[0],adj.shape[1])
        for i in range(len(adj_temp)):
            adj[i] = torch.sparse.FloatTensor(torch.LongTensor(adj_temp[i][0]).t(), 
                                                torch.FloatTensor(adj_temp[i][1]), 
                                                torch.Size(adj_temp[i][2])).to_dense()
        model = ChebyGCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    max_degree=3,
                    dropout=args.dropout)
    elif args.model == 'geo_gcn_cheby':
        model = GeoChebConv(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)
    elif args.model == 'geo_sage':
        model = GeoSAGEConv(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)
    elif args.model == 'geo_gat':
        model = GeoGATConv(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        # idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()


    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)

        # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # acc_val = accuracy(output[idx_val], labels[idx_val])
        if (epoch % 200) == 1:
            print('Epoch: {:04d}'.format(epoch+1))
            print(
                # 'Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'acc_train: {:.4f}'.format(acc_train.item()))
            test()
        #     'loss_val: {:.4f}'.format(loss_val.item()),
        #     'acc_val: {:.4f}'.format(acc_val.item()),
        #     'time: {:.4f}s'.format(time.time() - t))


    def test():
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print(
            # "Test set results for {}:".format(subject),
            "loss_test_= {:.4f}".format(loss_test.item()),
            "acc_test_= {:.4f}".format(acc_test.item()))
            # "test_labels= {}".format(labels[idx_test]))


    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()
