/*
 *
 *  GATO linear system solver
 *
 */

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <limits>

#include "src/admm_outer.cuh"


#include <sstream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>


namespace py = pybind11;

py::tuple main_call(std::vector<float> H,  std::vector<float>g, std::vector<float>A, std::vector<float> l , std::vector<float> u,  std::vector<float> x,  std::vector<float>lambda, std::vector<float>z, float rho, float sigma =1e-6, float tol =1e-3, int max_iters=1000){
	
	
	
	float *h_H, *h_g, *h_A, *h_l, *h_u, *h_x, *h_lambda, *h_z;
	h_H = H.data();
	h_g = g.data();
	h_A = A.data();
	h_l = l.data();
	h_u = u.data();
	h_x = x.data();
	h_lambda = lambda.data();
	h_z = z.data();

	admm_solve_outer<float>(h_H, h_g, h_A, h_l, h_u, h_x, h_lambda,  h_z, rho, sigma, tol, max_iters);

	py::list p_x;
    for(int i=0;i<NX;i++){
        p_x.append(h_x[i]);
    }

	py::list p_lambda;
    for(int i=0;i<NC;i++){
        p_lambda.append(h_lambda[i]);
    }
    py::list p_dz;
    for(int i=0;i<NC;i++){
        p_dz.append(h_z[i]);
    }

    py::tuple ans = py::make_tuple(p_x, p_lambda, p_dz);
    return ans;

}


PYBIND11_MODULE(gpu_library, m)
{
  m.def("admm_solve", &main_call, py::return_value_policy::move);
}
