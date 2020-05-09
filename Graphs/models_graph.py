import torch.nn as nn
import torch.nn.functional as F
from layers_graph import GraphConvolution, ChebyGraphConvolution

from torch_geometric.nn import ChebConv, SAGEConv, GATConv, SGConv, GCNConv
from torch_geometric.data import Data

import torch

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class ChebyGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, max_degree, dropout):
        super(ChebyGCN, self).__init__()

        nclass = int(nclass)
        self.gc1 = ChebyGraphConvolution(nfeat, nhid, max_degree)
        self.gc2 = ChebyGraphConvolution(nhid, nclass, max_degree)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GeoChebConv(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GeoChebConv,self).__init__()

        K = 3
        nclass = int(nclass)
        self.gc1 = ChebConv(nfeat, nhid, K)
        self.gc2 = ChebConv(nhid, nclass, K)
        self.dropout = dropout

    def forward(self, features, adj):
        data = Data(x=features, edge_index=adj._indices(), edge_attr=adj._values())
        x = F.relu(self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        return F.log_softmax(x, dim=1)


class GeoSAGEConv(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GeoSAGEConv,self).__init__()

        nclass = int(nclass)
        self.gc1 = SAGEConv(nfeat, nhid, normalize=True)
        self.gc2 = SAGEConv(nhid, nclass, normalize=True)
        self.dropout = dropout

    def forward(self, features, adj):
        data = Data(x=features, edge_index=adj._indices())
        x = F.relu(self.gc1(data['x'], edge_index=data['edge_index']))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index=data['edge_index'])
        return F.log_softmax(x, dim=1)


class GeoGATConv(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GeoGATConv,self).__init__()

        nclass = int(nclass)
        self.gc1 = GATConv(nfeat, nhid, heads=3, concat=False)
        self.gc2 = GATConv(nhid, nclass, heads=3, concat=False)
        self.dropout = dropout

    def forward(self, features, adj):
        data = Data(x=features, edge_index=adj._indices())
        x = F.relu(self.gc1(data['x'], edge_index=data['edge_index']))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index=data['edge_index'])
        return F.log_softmax(x, dim=1)


class GeoSGConv(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GeoSGConv,self).__init__()

        nclass = int(nclass)
        self.gc1 = SGConv(nfeat, nhid)
        self.gc2 = SGConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, features, adj):
        data = Data(x=features, edge_index=adj._indices(), edge_attr=adj._values())
        x = F.relu(self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x.type(torch.float32), edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        return F.log_softmax(x, dim=1)


class GeoGCNConv(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GeoGCNConv,self).__init__()

        nclass = int(nclass)
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, features, adj):
        data = Data(x=features, edge_index=adj._indices(), edge_attr=adj._values())
        x = F.relu(self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x.type(torch.float32), edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        return F.log_softmax(x, dim=1)