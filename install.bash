git submodule init
git submodule update
cd build
cmake -DCUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so ..
make
export PYTHONPATH=$PWD:$PYTHONPATH
cd ..
