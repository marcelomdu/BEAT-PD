from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_graph import load_data, accuracy, chebyshev_polynomials, threshold_adj, cn_test
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
parser.add_argument('--cn_type', type=int, default=13,
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
# seed = np.random.randint(100)
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

for subject in subjects_list:
    
    for i in range(4,14):
        cn_test(path=path,subject=str(subject),label=label,cn_type=i,ft_type=ft_type)
    
