import os
import warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '32'

import time
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph

import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

class RicciFlowTrainer:
    def __init__(self,affinity='nearest_neighbors',gamma=None):
        self.affinity = affinity
        self.gamma = gamma

    def get_weight_matrix(self,X,n_neighbors=10):
        """Calculate the affinity distance matrix from data
        """
        if self.affinity == "nearest_neighbors":
                n_neighbors_ = (
                    n_neighbors
                    if n_neighbors is not None
                    else max(int(X.shape[0] / 10), 1)
                )
                affinity_matrix_ = kneighbors_graph(
                    X, n_neighbors_, include_self=True, mode='distance',p=2
                )
                # currently only symmetric affinity_matrix supported
                affinity_matrix_ = 0.5 * (
                    affinity_matrix_ + affinity_matrix_.T
                )
                return affinity_matrix_
        if self.affinity == "rbf":
            gamma_ = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
            affinity_matrix_ = rbf_kernel(X, gamma=gamma_)
        return affinity_matrix_

    def matrix_prune(self,A, prune_ratio=0.1):
        # prune the top big values in the distance matrix
        B = A.flatten()
        non_zero_num = np.count_nonzero(B)
        top_big_thre = int(prune_ratio * non_zero_num)
        if top_big_thre != 0:
            top_big_thre_value = np.sort(B[np.nonzero(B)])[-top_big_thre]
        else:
            return A
        A[A >= top_big_thre_value] = 0.0
        return A
    def update_weight_Ollivier(self, A,lr=0.8,iterations=7,alpha=0.4,prune_ratio=0.0,save_nbr_matrix=False,dataset_name=''):
        #weight_matrix is distance matrix
        if iterations==0:
            return A
        if isinstance(A, np.ndarray):
            A = A
        else:
            A = A.toarray()

        # convert distance matrix to similarity matrix
        A_arr = A.copy().astype(float)
        mask = A_arr != 0.0
        A_arr[mask] = 1 / A_arr[mask]
        A = A_arr

        # update the similarity matrix with Ollivier-Ricci flow
        start_time = time.time()
        for i in range(iterations):
            nx_graph = nx.from_numpy_matrix(A)
            orc = OllivierRicci(nx_graph, alpha=alpha, verbose="ERROR")
            orc.compute_ricci_curvature()
            nx_graph_ricci_curv = np.asarray(nx.to_numpy_matrix(orc.G, weight='ricciCurvature'))
            A = A * (1 - lr * nx_graph_ricci_curv)
            if save_nbr_matrix:
                np.savez_compressed(
                        f'./matrixs/{dataset_name}_Ollivier_lr{lr}_alpha{alpha}_iter{i + 1}_prune_ratio{prune_ratio}.npz',
                        matrix=self.norm(A))
        end_time = time.time()
        cost = end_time - start_time
        print('iter times', iterations ,'Ollivier cost:', cost)

        # convert similarity matrix to distance matrix
        A_arr = A.copy().astype(float)
        mask = A_arr != 0.0
        A_arr[mask] = 1 / A_arr[mask]
        A = A_arr

        return self.matrix_prune(self.norm(A),prune_ratio=prune_ratio)

    def update_weight_Formann(self, A,lr=0.8,iterations=7,prune_ratio=0.0,save_nbr_matrix=False,dataset_name=''):
        #weight_matrix is distance matrix
        if iterations==0:
            return A
        if isinstance(A, np.ndarray):
            A = A
        else:
            A = A.toarray()

        # convert distance matrix to similarity matrix
        A_arr = A.copy().astype(float)
        mask = A_arr != 0.0
        A_arr[mask] = 1 / A_arr[mask]
        A = A_arr

        # update the similarity matrix with Forman-Ricci flow
        start_time = time.time()
        for i in range(iterations):
            nx_graph = nx.from_numpy_matrix(A)
            frc=FormanRicci(nx_graph, weight="weight", verbose="ERROR")
            frc.compute_ricci_curvature()
            nx_graph_ricci_curv = np.asarray(nx.to_numpy_matrix(frc.G, weight='formanCurvature'))
            A=A * (1 -lr * nx_graph_ricci_curv)
            if save_nbr_matrix:
                np.savez_compressed(
                        f'./matrixs/{dataset_name}_Formann_lr{lr}_iter{i + 1}_prune_ratio{prune_ratio}.npz',
                        matrix=self.norm(A))

        end_time = time.time()
        cost = end_time - start_time
        print('Formann cost:', cost)

        # convert similarity matrix to distance matrix
        A_arr = A.copy().astype(float)
        mask = A_arr != 0.0
        A_arr[mask] = 1 / A_arr[mask]
        A = A_arr
        return self.matrix_prune(self.norm(A),prune_ratio=prune_ratio)

    def norm(self,A):
        # normalize the matrix to [0,1] in the global view
        A_arr = A.astype(float)
        _range = np.max(A_arr) - np.min(A_arr)
        norm = (A_arr - np.min(A_arr)) / _range
        return norm
