import subprocess
import gpu_library
import numpy as np
import time
from scipy import sparse
print("")
print("------------------LINSYS OUTPUT----------------------------------")
print("")
G_row =  [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]
G_col = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]
G_val = [ 1. ,   1. ,   0.1,   1. ,   1. ,   0.1,   1. ,   1. ,   0.1,   1. ,   1. ,   0.1, 100., 100.]

C_row = [ 0,  1,  2,  5,  9, 12, 16, 19, 23, 26, 30]
C_col = [0,  1,  0,  1,  3,  0,  1,  2,  4,  3,  4,  6,  3,  4,  5,  7,  6,  7,  9,  6,  7,  8, 10, 9, 10, 12,  9, 10, 11, 13]
C_val = [ 1.   ,  1.   , -1.   , -0.1  ,  1.   ,  0.981, -1.   , -0.1  ,  1.   , -1.   , -0.1  , 1.   ,  0.981, -1.   , -0.1  ,  1.   , -1.   , -0.1  ,  1.   ,  0.981, -1.   , -0.1  , 1.   , -1.   , -0.1  ,  1.   ,  0.981, -1.   , -0.1  ,  1.]

g_val = [  -3.1416,   0.    ,   0.    ,  -3.1416,   0.    ,   0.    ,  -3.1416,   0.    ,   0.    ,  -3.1416,   0.    ,   0.    , -314.159 ,   0.]
c_val = [0,0,0,0,0,0,0,0,0,0]
testiters = 10
exit_tol = 1e-6
max_iters = 10
warm_start = False
input_lambda = [0.,0.,0.,0.,0.,0.,0.,0.,0., 0.]
rho = .001
l, dz = gpu_library.linsys_solve(G_row, G_col, G_val, C_row, C_col, C_val, g_val, c_val, input_lambda, testiters, exit_tol, max_iters, warm_start, rho)


G_csr = sparse.csr_matrix((G_val, G_col, G_row))
C_csr = sparse.csr_matrix((C_val, C_col, C_row))
G = G_csr.todense()
C = C_csr.todense()
A = np.block([[G, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
gamma = np.block([[np.array([g_val]).T],[ np.array([c_val]).T]])
x = (np.linalg.inv(A).dot(gamma))
x_gato = np.block([[np.array([dz]).T],[ np.array([l]).T]])

assert(np.allclose(x_gato, x , rtol=1,atol=0.01))
print("")
print("-----------------------------------------------------------------")
print("Test passed")