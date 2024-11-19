import torch.nn as nn
from model import Model, SimplePrompt, GPFplusAtt, Projection
from utils import *
import random
from pretrain import traingrace
import torch.nn.functional as F
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))
os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
device = torch.device("cuda")
parser = argparse.ArgumentParser(description='Unified Neighborhood Prompt for Graph Anomaly Detection')
parser.add_argument('--dataset', type=str, default='Facebook')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--edge_drop_prob', type=float, default=0.2)
parser.add_argument('--feat_drop_prob', type=float, default=0.3)
parser.add_argument('--lamda', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--unifeat', type=int, default=8)
parser.add_argument('--numprompts', type=int, default=10)
args = parser.parse_args()

dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Load and preprocess data
def loaddata(dataset, args, device):
    adj, features,  ano_label, str_ano_label, attr_ano_label = load_mat(dataset)
    adj = adj.astype(np.float32)
    features = features.todense()
    features = torch.FloatTensor(features)   
    features = x_svd(features, args.unifeat)

    bn = nn.BatchNorm1d(features.shape[1], affine=False)
    features = bn(features)

    diag_adj = adj.diagonal()>0
    if diag_adj.all():
        adj_withloop_won = adj
        adj_woself = adj - sp.eye(adj.shape[0])
    else:
        adj_withloop_won = adj + sp.eye(adj.shape[0])
        adj_woself = adj
    adj_withloop = normalize_adj(adj_withloop_won)
    adj_withloop = sparse_mx_to_torch_sparse_tensor(adj_withloop)
    adj_woself = normalize_adj(adj_woself)
    adj_woself = sparse_mx_to_torch_sparse_tensor(adj_woself)
    
    ano_label = torch.FloatTensor(ano_label)
    ano_label = ano_label.to(device)
    adj_withloop = adj_withloop.to(device)
    adj_woself = adj_woself.to(device)
    features = features.to(device)
    return adj_withloop_won, adj_withloop, adj_woself, features, ano_label

traindatasets = ['Facebook']
targdataset = ['Amazon', 'Reddit']

adj_withloop_won_train = []
adj_withloop_train = []
adj_woself_train = []
features_train = []
ano_label_train = []
for dataset in traindatasets:
    adj_withloop_won, adj_withloop, adj_woself, features, ano_label = loaddata(dataset, args, device)
    adj_withloop_won_train.append(adj_withloop_won)
    adj_withloop_train.append(adj_withloop)
    adj_woself_train.append(adj_woself)
    features_train.append(features)
    ano_label_train.append(ano_label)
    

all_aucs = []
all_aps = []

for _ in range(1):

    model = Model(args.unifeat, args.embedding_dim, 'prelu')
    model = model.to(device)
    traingrace(model, adj_withloop_won_train, adj_withloop_train, features_train, args, device)

    model.eval()
    if args.numprompts < 2:
        prompts = SimplePrompt(args.unifeat)
    else:
        prompts = GPFplusAtt(args.unifeat, args.numprompts)
    proj = Projection(args.embedding_dim)
    prompts = prompts.to(device)
    proj = proj.to(device)

    all_params = list(prompts.parameters()) + list(proj.parameters())
    optimiser_prompt_proj = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        for dataset in range(len(traindatasets)):
            prompts.train()
            proj.train()
            optimiser_prompt_proj.zero_grad()

            adj_woself = adj_woself_train[dataset]
            features = features_train[dataset]
            ano_label = ano_label_train[dataset]

            modified_feature = prompts.add(features)
            node_emb_nei = model(modified_feature, adj_woself)
            node_emb_mlp = model(modified_feature, None)
            node_emb_nei = proj(node_emb_nei)
            node_emb_mlp = proj(node_emb_mlp)

            loss = completionloss(node_emb_nei, node_emb_mlp, ano_label)
            
            loss.backward()
            optimiser_prompt_proj.step()

    ##### Test on Target Datasets
    prompts.eval()
    proj.eval()
    aucs = []
    aps = []
    for dataset in targdataset:
        _, _, adj_woself, features, ano_label = loaddata(dataset, args, device)

        modified_feature_tar = prompts.add(features)
        node_emb_nei = model(modified_feature_tar, adj_woself)
        node_emb_mlp = model(modified_feature_tar, None)
        node_emb_nei = proj(node_emb_nei)
        node_emb_mlp = proj(node_emb_mlp)

        completion_message = completionsim(node_emb_mlp, node_emb_nei)
        completion_auc, completion_AP = evaluate(completion_message, ano_label.cpu().numpy())
        aucs.append(completion_auc)
        aps.append(completion_AP)
        print('{} -> {} AUC:{:.4f} AP{:.4f}'.format(" ".join(traindatasets), dataset, completion_auc, completion_AP))

    all_aucs.append(aucs)
    all_aps.append(aps)

all_aucs, all_aps = np.array(all_aucs), np.array(all_aps)
mean_auc, std_auc = np.mean(all_aucs, 0), np.std(all_aucs, 0)
mean_ap, std_ap = np.mean(all_aps, 0), np.std(all_aps, 0)

for i, dataset in enumerate(targdataset):
    with open(f'results/{args.dataset}.txt','a') as f:
        f.write('\n Averaged {} -> {} AUC:{:.4f}$_{{\\pm {:.3f}}}$ AP:{:.4f}$_{{\\pm {:.3f}}}$\n'.format(args.dataset, dataset, mean_auc[i], std_auc[i], mean_ap[i], std_ap[i]))
 
