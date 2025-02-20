import torch
import torch.nn as nn
import math
import scipy.sparse as sp
import sys
import os
sys.path.append(
    '%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(
        os.path.realpath(__file__)))
from pytorch_util import weights_init, gnn_spmm # noqa
import torch.nn.functional as F
from torch.autograd import Variable
from util import cmd_args
class Pool(nn.Module):
    def __init__(self, k, lr=1, alpha=0.5,latent_dim=[48, 48],device='cpu',number_iterations=5):
        super(Pool, self).__init__()

        self.latent_dim = latent_dim
        self.latent_dim.append(k)
        self.k = k 
        self.lr = lr
        self.alpha = alpha
        self.number_iterations = number_iterations
        self.softmax = nn.Softmax(dim=None)
        self.w_filter = Variable(torch.eye(k)).float().to(device=device)
        self.w_compat = Variable(-1*torch.eye(k)).float().to(device=device)

    def forward(self, A, X, U, graph):
        ''' Use GCNs to obtain u(x) '''
        device = A.device
        #load adjusted weight via curvature
        A=torch.load(f'./weights/{cmd_args.data}/graph{graph.order}_lr{self.lr}_alpha{self.alpha}_iter{self.number_iterations}.pt')
        A=A.to(device=device)
        q_values=U
        n2n_sp=A
        L = F.softmax(q_values, dim=-1) #[b,n,k]
        L = L.to(device=device)
        L_onehot = L
        L_onehot_T = torch.transpose(L_onehot, -2, -1)#[b,k,n]
        X_out = torch.matmul(L_onehot_T, X) #[b,k,d]

        A_out0 = torch.matmul(L_onehot_T, A)
        A_out = torch.matmul(A_out0,L_onehot)
        
        if self.training:
            n2n_sp_out0 = torch.matmul(L_onehot_T, n2n_sp)
            n2n_sp_out = torch.matmul(n2n_sp_out0, L_onehot)

            D = torch.ones([A.shape[0], 1]).to(A)
            D_Aout0 = torch.diag_embed(torch.matmul(A, D).squeeze(1))
            D_A0 = torch.diag_embed(torch.matmul(n2n_sp, D).squeeze(1))

            D_Aout = torch.matmul(torch.matmul(L_onehot_T, D_Aout0), L_onehot)
            D_A = torch.matmul(torch.matmul(L_onehot_T, D_A0), L_onehot)
            pt_p = torch.matmul(L_onehot_T, L_onehot)
            f_loss = torch.norm(pt_p / torch.norm(pt_p) - torch.eye(L_onehot.shape[1]).to(pt_p) / math.sqrt(L_onehot.shape[1]))
            ricci_pool_loss = A_out.trace() / D_Aout.trace() - n2n_sp_out.trace() / D_A.trace() + f_loss
            return X_out, A_out, ricci_pool_loss
        return X_out, A_out

    def sparse_to_dense(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj.to_dense().cpu().numpy()
        adj = sp.coo_matrix(adj).tocoo()
        return torch.FloatTensor(adj.todense()).cuda()

