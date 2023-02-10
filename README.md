# gato-python

Python interface for a GPU-accelerated linear system solver. 


Originally based on https://github.com/torstem/demo-cuda-pybind11

Template generated from https://github.com/PWhiddy/pybind11-cuda

# Prerequisites

Cuda installed in /usr/local/cuda 

Python 3.6 or greater 

Cmake 3.6 or greater 

# To build 

```source install.bash``` 

Test it with 
```python3 test_linsys_solve.py```

# Example

Make sure you set the following variable according to the problem in "include/gato_defines.h"
```
#define STATE_SIZE   2
#define CONTROL_SIZE 1
#define KNOT_POINTS 5
```
And, then run the build command
```source install.bash``` 


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
input_lambda = [0.,0.,0.,0.,0.,0.,0.,0.,0., 0., 0.]
l, dz = gpu_library.linsys_solve(G_row, G_col, G_val, C_row, C_col, C_val, g_val, c_val, input_lambda, testiters, exit_tol, max_iters, warm_start)
```
