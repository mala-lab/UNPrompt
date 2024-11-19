import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *


def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def mask_edge(adj_withloop_won, drop_prob):
    adj_withloop_won = adj_withloop_won.tocoo()
    num_edges = adj_withloop_won.nnz
    size = adj_withloop_won.shape
    edge_delete = np.random.choice(num_edges, int(drop_prob*num_edges), replace=False)
    row, col = adj_withloop_won.row, adj_withloop_won.col
    not_equal = row[edge_delete] != col[edge_delete]
    edge_delete = edge_delete[not_equal]
    keep_mask = np.ones(num_edges, dtype=bool)
    keep_mask[edge_delete] = False

    newdata = adj_withloop_won.data[keep_mask]
    newrow = adj_withloop_won.row[keep_mask]
    newcol = adj_withloop_won.col[keep_mask]
    adj_withloop_won = sp.coo_matrix((newdata, (newrow, newcol)), shape=size)
    
    adj_aug = normalize_adj(adj_withloop_won)
    adj_aug = sparse_mx_to_torch_sparse_tensor(adj_withloop_won)
    return adj_aug


class ModelGrace(nn.Module):
    def __init__(self, model, num_hidden, num_proj_hidden, tau=0.5):
        super(ModelGrace, self).__init__()
        self.model = model
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, features, adj):
        output = self.model(features, adj)
        Z = F.elu(self.fc1(output))
        Z = self.fc2(Z)
        return Z

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1, z2, batch_size):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1[mask]))
            between_sim = f(self.sim(z1[mask], z2[mask]))
            losses.append(-torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())))
            torch.cuda.empty_cache()
        return torch.cat(losses)

    def loss(self, h1, h2, batch_size):
        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        ret = (l1 + l2) * 0.5
        return ret


def traingrace(model, adj_withloop_won_train, adj_withloop_train, features_train, args, device):
    batch_size = None
    drop_edge_prob = args.edge_drop_prob
    drop_feature_prob = args.feat_drop_prob
    epochs = 200
    lr = 1e-3
    gracemodel = ModelGrace(model, num_hidden=args.embedding_dim, num_proj_hidden=2*args.embedding_dim)
    optimizer = torch.optim.Adam(gracemodel.parameters(), lr=lr, weight_decay=1e-5)
    gracemodel = gracemodel.to(device)
    for epoch in range(epochs):
        for dataset in range(len(features_train)):
            gracemodel.train()
            optimizer.zero_grad()

            features = features_train[dataset]
            adj_withloop_won = adj_withloop_won_train[dataset]
            adj_withloop = adj_withloop_train[dataset]

            feat_aug = drop_feature(features, drop_feature_prob)
            if drop_edge_prob > 0:
                adj_aug = mask_edge(adj_withloop_won, drop_edge_prob)
                adj_aug = adj_aug.to(device)
            else:
                adj_aug = adj_withloop

            Z1 = gracemodel(features, adj_withloop)
            Z2 = gracemodel(feat_aug, adj_aug)
            loss = gracemodel.loss(Z1, Z2, batch_size=batch_size)
            loss = loss.mean()
            loss.backward()
            optimizer.step()