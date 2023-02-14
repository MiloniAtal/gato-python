import gpu_library
import numpy as np
import time
from scipy import sparse
import sys, os
sys.path.append("/home/a2rlab/ppcg/TrajoptReference_Dev/")
print(sys.path)
from testHelpers import getKKT
from testIIWA import getIIWA

N = 50
trajoptReference, x, u, xs, xg, dt = getIIWA(50)
G, g, C, c = getKKT(trajoptReference, x, u, xs, xg, dt)

G_csr = sparse.csr_matrix(G)
G_val = G_csr.data()
G_row = G_csr.indptr()
G_col = G_csr.indices()