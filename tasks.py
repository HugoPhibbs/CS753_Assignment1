import scipy as sp
import numpy as np


def read_adjacency_dataset(filename, N):
    with open(filename) as f:
        lines = f.readlines()

    M = sp.sparse.lil_matrix((N, N))

    for l in lines:
        edge_indices = l.strip("\n").split('\t')
        i, j = int(edge_indices[0]), int(edge_indices[1])
        M[i, j] = 1
        M[j, i] = 1

    M_csr = M.tocsr()

    return M_csr


def csr_mat_to_col_stochastic(M_csr):
    col_sums = np.array(M_csr.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1
    M_csr.multiply(1/col_sums)

    return M_csr
