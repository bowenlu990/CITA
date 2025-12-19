import torch.nn as nn
import torch
import math
import random
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops, degree, add_remaining_self_loops, negative_sampling
from data_utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GATv2Conv, SAGEConv
from sklearn.feature_selection  import mutual_info_classif

def gcn_conv(x, edge_index):
    N = x.shape[0]

    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    value = torch.ones_like(row) * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(N, N))
    return adj # [N, D]


class generate_augmented_adj(nn.Module):
    def __init__(self, n, alpha, num_edges):
        super(generate_augmented_adj, self).__init__()
        self.n=n

        self.edge_perturb=Parameter(torch.FloatTensor(num_edges))

        self.tanh=nn.Tanh()
        self.alpha=alpha
        self.num_edges=num_edges
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.edge_perturb.size(0))
        self.edge_perturb.data.uniform_(-stdv, stdv)


    def forward(self, x, adj):
        gcn_adj = gcn_conv(x, adj)

        row, col, orig_val = gcn_adj.coo()

        perturbation = self.alpha * self.tanh(self.edge_perturb)


        new_val = orig_val + perturbation


        augmented_adj = SparseTensor(row=row, col=col, value=new_val, sparse_sizes=(self.n, self.n))


        return augmented_adj


class GraphConvolutionBase(nn.Module):

    def __init__(self, in_features, out_features, residual=False):
        super(GraphConvolutionBase, self).__init__()
        self.residual = residual
        self.in_features = in_features

        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if self.residual:
            self.weight_r = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_r.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, augment_adj=None):
        if augment_adj is not None:
            adj = augment_adj
        else:
            adj = gcn_conv(x, adj)
        hi=matmul(adj, x)

        output = torch.mm(hi, self.weight)
        if self.residual:
            output = output + torch.mm(x, self.weight_r)
        return output

class layer(nn.Module):

    def __init__(self, in_features, out_features, K, n, args, residual=True, device=None):
        super(layer, self).__init__()

        self.out_features = out_features
        self.residual = residual

        self.weights1 = Parameter(torch.FloatTensor(K, in_features*2, out_features))
        self.leakyrelu = nn.LeakyReLU()
        self.weights2 = nn.Parameter(torch.zeros(K, in_features, out_features))
        self.a = nn.Parameter(torch.zeros(K, 2 * out_features, 1))

        self.K = K

        self.device = device

        self.tau = args.tau
        self.reset_parameters()
        self.beta = args.beta


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weights1.data.uniform_(-stdv, stdv)
        self.weights2.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def specialspmm(self, adj, spm, size, h):
        row, col, val=adj.coo()

        adj = SparseTensor(row=row, col=col, value=spm, sparse_sizes=size)

        return matmul(adj, h)

    def forward(self, x, adj, aug_adj, r):
        hi = matmul(aug_adj, x)

        h1 = torch.cat([hi, x], 1)           #
        h2 = h1.unsqueeze(0).repeat(self.K, 1, 1)
        outputs = torch.matmul(h2, self.weights1)
        att=torch.mean(outputs, dim=(1,2), keepdim=True)
        att_softmax=F.softmax(att, dim=0)
        outputs=att_softmax*outputs
        outputs1 = outputs.transpose(1, 0)

        hi = x.unsqueeze(0).repeat(self.K, 1, 1)
        h = torch.matmul(hi, self.weights2)
        N = x.size()[0]
        adj, _ = remove_self_loops(adj)
        adj, _ = add_self_loops(adj, num_nodes=N)
        row, col, val = aug_adj.coo()
        edge_h = torch.cat((h[:, row, :], h[:, col, :]), dim=2)
        logits = self.leakyrelu(torch.matmul(edge_h, self.a)).squeeze(2)

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        edge_e = torch.exp(logits - logits_max)
        outputs = []
        eps = 1e-8
        for k in range(self.K):
            edge_e_k = edge_e[k, :]  # [E]

            e_expsum_k=matmul(SparseTensor(row=row, col=col, value=edge_e_k, sparse_sizes=torch.Size([N,N])), torch.ones(N, 1).cuda())+eps
            assert not torch.isnan(e_expsum_k).any()


            hi_k=matmul(SparseTensor(row=row, col=col, value=edge_e_k, sparse_sizes=torch.Size([N,N])), h[k])
            hi_k = torch.div(hi_k, e_expsum_k)  # [N, D]
            outputs.append(hi_k)

        outputs2 = torch.stack(outputs, dim=1)  # [N, K, D]



        outputs=self.beta*outputs1+ self.beta*outputs2
        theta = r.unsqueeze(2).repeat(1, 1, self.out_features)
        output = torch.sum(torch.mul(theta, outputs), dim=1)

        if self.residual:
            output = output + x

        return output

class CausalFrontNet(nn.Module):
    def __init__(self, d, c, n, num_edges, args, device):
        super(CausalFrontNet, self).__init__()
        self.n = n
        self.convs = nn.ModuleList()
        for _ in range(args.num_layers):
            self.convs.append(layer(args.hidden_channels, args.hidden_channels, args.K, n=self.n, args=args, residual=True,device=device ))
        self.fcs = nn.ModuleList()
        self.weight=nn.Parameter(torch.FloatTensor(args.K*args.num_layers, args.K))
        self.weights=nn.Parameter(torch.FloatTensor(args.hidden_channels*args.num_layers, args.hidden_channels))
        self.generate_augmented_adj = generate_augmented_adj(self.n, args.alpha, num_edges)

        self.fcs.append(nn.Linear(d, args.hidden_channels))

        self.fcs.append(nn.Linear(args.hidden_channels, c))
        self.env_enc = nn.ModuleList()
        for _ in range(args.num_layers):
            self.env_enc.append(GraphConvolutionBase(args.hidden_channels, args.K, residual=True))
        self.act_fn = nn.ReLU()
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.tau = args.tau
        self.K=args.K

        self.device = device
        self.hidden=args.hidden_channels

        self.num_edges=num_edges
        self.weight_sim=Parameter(torch.FloatTensor(args.K, args.K))
        self.K=args.K
        self.sage1 = SAGEConv(in_channels=args.hidden_channels, out_channels=args.hidden_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        for enc in self.env_enc:
            enc.reset_parameters()
        self.sage1.reset_parameters()

        self.generate_augmented_adj.reset_parameters()
        stdv = 1. / math.sqrt(self.K*self.num_layers)
        self.weight.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.hidden * self.num_layers)
        self.weights.data.uniform_(-stdv1, stdv1)
        stdv_sim = 1. / math.sqrt(self.K)
        self.weight_sim.data.uniform_(-stdv_sim, stdv_sim)


    def forward(self, x, adj, training=False):
        self.training = training

        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x))
        h=self.sage1(h,adj)

        aug_adj = self.generate_augmented_adj(h, adj)

        list1=[]
        list2=[]
        reg = 0
        for i,con in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            logit = self.env_enc[i](h, adj, aug_adj)

            if self.training:

                sim=F.cosine_similarity(logit, logit, dim=1).unsqueeze(dim=1).repeat(1, logit.shape[1])
                ZA=torch.matmul(sim, self.weight_sim)
                ZA= F.relu(ZA)
                logit=ZA * logit
                theta = F.gumbel_softmax(logit, tau=self.tau, dim=-1)
                reg += self.struct_loss(theta, logit)
            else:

                theta = F.softmax(logit, dim=-1)

            list1.append(theta)
            h = self.act_fn(con(h, adj, aug_adj, theta))
            list2.append(h)
        r=torch.cat(list1, dim=-1)
        h1=torch.cat(list2, dim=-1)
        r=torch.matmul(r, self.weight)
        h=torch.matmul(h1, self.weights)
        h=self.act_fn(self.convs[self.num_layers-1](h, adj, aug_adj, r))
        h = F.dropout(h, self.dropout, training=self.training)
        out = self.fcs[-1](h)                   #out:[N,c]
        if self.training:
            return out, reg / self.num_layers
        else:
            return out

    def reg_loss(self, logit, args):

        sim=F.cosine_similarity(logit, logit, dim=1).unsqueeze(1).repeat(1, logit.size(1))
        log_pi = sim/args.tau - torch.logsumexp(sim/args.tau, dim=-1, keepdim=True)
        return torch.mean(torch.sum(log_pi, dim=1))

    def struct_loss(self, z, logit):
        log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, logit.size(1)) #logit:[N,K]
        return torch.mean(torch.sum(
            torch.mul(z, log_pi), dim=1))    #z:[N,K]
    def sup_loss_calc(self, y, pred, criterion, args):
        if args.dataset in ('twitch', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss

    def label_smoothing(self, y_one_hot, epsilon=0.1):
        y_smooth = (1 - epsilon) * y_one_hot + epsilon / y_one_hot.shape[1]
        return y_smooth

    def loss_compute(self, d, criterion, args):
        logits, struct_loss = self.forward(d.x, d.edge_index, training=True)
        sup_loss = self.sup_loss_calc(d.y[d.train_idx], logits[d.train_idx], criterion, args)

        loss = sup_loss + args.lamda1 * struct_loss + args.lamda2 * self.reg_loss(logits, args)
        return loss