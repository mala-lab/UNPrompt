import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import dgl

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
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


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj_tensor(raw_adj):
    adj = raw_adj
    row_sum = torch.sum(adj, 0)
    r_inv = torch.pow(row_sum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    adj = torch.mm(adj, torch.diag_embed(r_inv))
    adj = torch.mm(torch.diag_embed(r_inv), adj)
    return adj

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = sparse_mx.shape
    return torch.sparse_coo_tensor(indices, values, shape)

def normalize_score(ano_score):
    ano_score = ((ano_score - np.min(ano_score)) / (np.max(ano_score) - np.min(ano_score)))
    return ano_score

def x_svd(data, out_dim):
    assert data.shape[-1] >= out_dim
    U, S, _ = torch.linalg.svd(data)
    newdata= torch.mm(U[:, :out_dim], torch.diag(S[:out_dim]))
    return newdata


def load_mat(dataset):
    """Load .mat dataset."""
    data = sio.loadmat("./Datasets/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None
    return adj, feat, ano_labels, str_ano_labels, attr_ano_labels


def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_array(adj)
    dgl_graph = dgl.from_networkx(nx_graph)
    return dgl_graph


def completionloss(feature1, feature2, ano_label):
    feature1 = feature1 / torch.norm(feature1, dim=-1, keepdim=True)
    feature2 = feature2 / torch.norm(feature2, dim=-1, keepdim=True)
    diff = -torch.sum(feature1*feature2, dim=1)
    modified = torch.where(ano_label == 0, diff, -diff)
    loss = torch.mean(modified)
    return loss


def completionsim(feature1, feature2):
    feature1 = feature1 / torch.norm(feature1, dim=-1, keepdim=True)
    feature2 = feature2 / torch.norm(feature2, dim=-1, keepdim=True)
    dist = torch.sum(feature1*feature2, dim=1)
    dist = dist.detach().cpu().numpy()
    return dist

def evaluate(message, ano_label, str_ano_label=None, attr_ano_label=None):
    score = 1-normalize_score(message)
    auc = roc_auc_score(ano_label, score)
    AP = average_precision_score(ano_label, score, average='macro', pos_label=1, sample_weight=None)

    if str_ano_label is not None:
        sa_auc = roc_auc_score(str_ano_label, score)
        sa_AP = average_precision_score(str_ano_label, score, average='macro', pos_label=1, sample_weight=None)
        print('Structural: AUC: {:.4f} AP:{:.4f}'.format(sa_auc, sa_AP))
    if attr_ano_label is not None:
        aa_auc = roc_auc_score(attr_ano_label, score)
        aa_AP = average_precision_score(attr_ano_label, score, average='macro', pos_label=1, sample_weight=None)
        print('Context: AUC:{:.4f} AP:{:.4f}'.format(aa_auc, aa_AP))
    return auc, AP


def reg_edge(emb, adj):
    emb = emb / torch.norm(emb, dim=-1, keepdim=True)
    sim_u_u = torch.mm(emb, emb.T)
    adj_inverse = (1 - adj)
    sim_u_u = sim_u_u * adj_inverse
    sim_u_u_no_diag = torch.sum(sim_u_u, 1)
    row_sum = torch.sum(adj_inverse, 1)
    r_inv = torch.pow(row_sum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    sim_u_u_no_diag = sim_u_u_no_diag * r_inv
    loss_reg = torch.sum(sim_u_u_no_diag)
    return loss_reg

def reg_sim(feature1, feature2):
    feature1 = feature1 / torch.norm(feature1, dim=-1, keepdim=True)
    feature2 = feature2 / torch.norm(feature2, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature2, feature1.T)
    sim_matrix = sim_matrix.sum(1) - sim_matrix.diag()
    non_diag_sim = torch.mean(sim_matrix)
    return non_diag_sim


