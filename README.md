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
STATE_SIZE  =
CONTROL_SIZE = 
KNOT_POINTS =
```

# Example

The build command would look like this for the below test case of Pendulum with 5 Knotpoints:
```source install.bash 2 1 5``` 

Then the example can be run:
```

```
