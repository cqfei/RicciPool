import torch

from util import load_data,cmd_args
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci,_compute_ricci_curvature_edges
import time

# cmd_args.data='DD'
cmd_args.data='PROTEINS'
number_iterations=5
lrs=[1.0,0.9,0.8,0.7,0.6,0.5,0.4]
alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7]
lrs=[1.0]
alphas=[0.5]
def cal_curvature():
    import sys
    import os
    sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(
        os.path.realpath(__file__)))
    from s2v_lib import S2VLIB
    g_list = load_data()
    print('number of graphs: ',len(g_list))
    g_adj_list=[]
    for i in range(len(g_list)):
        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField([g_list[i]])
        g_adj_list.append(n2n_sp.to_dense().cpu().numpy())
    print('start calculating curvatures')
    start_time=time.time()
    for i in range(len(g_adj_list)):
        # print(i)
        if i%50==0:
            print('have processed %d graphs'%(i+1))
            print(f'{i+1} graphs Cost: ', time.time() - start_time)
        A = g_adj_list[i]
        for lr in lrs:
            for alpha in alphas:
                for j in range(number_iterations):
                    # if os.path.exists(f'./curvatures/{cmd_args.data}/graph{i}_lr{lr}_alpha{alpha}_iter{j+1}.pt'):
                    #     continue
                    # if os.path.exists(f'./weights/{cmd_args.data}/graph{i}_lr{lr}_alpha{alpha}_iter{j+1}.pt'):
                    #     continue

                    nx_graph = nx.from_numpy_array(A)
                    orc = OllivierRicci(nx_graph, alpha=alpha, verbose="INFO")
                    orc.compute_ricci_curvature()
                    nx_graph_ricci_curv = np.asarray(nx.to_numpy_array(orc.G, weight='ricciCurvature'))
                    A = A * (1 - lr * nx_graph_ricci_curv)
                    # if not os.path.exists(f'./curvatures/{cmd_args.data}'):
                    #     os.mkdir(f'./curvatures/{cmd_args.data}')
                    # if not os.path.exists(f'./weights/{cmd_args.data}'):
                    #     os.mkdir(f'./weights/{cmd_args.data}')
                    #
                    # torch.save(torch.Tensor(nx_graph_ricci_curv),f'./curvatures/{cmd_args.data}/graph{i}_lr{lr}_alpha{alpha}_iter{j+1}.pt',)
                    # torch.save(torch.Tensor(A),f'./weights/{cmd_args.data}/graph{i}_lr{lr}_alpha{alpha}_iter{j+1}.pt',)
    print('Cost: ',time.time()-start_time)


if __name__ == '__main__':
    cal_curvature()