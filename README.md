# gato-python

Python interface for a GPU-accelerated linear system solver. 


Originally based on https://github.com/torstem/demo-cuda-pybind11

Template generated from https://github.com/PWhiddy/pybind11-cuda

# Prerequisites

Cuda installed in /usr/local/cuda 

Python 3.6 or greater 

Cmake 3.6 or greater 

# To build 

```source install.bash [STATE_SIZE] [CONTROL_SIZE] [KNOT_POINTS] [NC] [NX]``` 

Make sure to pass the variable according to the problem when calling the build command. If you donot specify these numbers, the below default values will be used
```
STATE_SIZE  = 14
CONTROL_SIZE = 7
KNOT_POINTS = 50
```

# Example

The build command would look like this for the below test case of Pendulum with 5 Knotpoints:
```source install.bash 2 1 5``` 

Then the example can be run:
```
import gpu_library
import numpy as np
import time

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

```
