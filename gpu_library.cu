/*
 *
 *  GATO linear system solver
 *
 */

#include <iostream>
#include <stdio.h>
#include <assert.h>

#include "include/gato_defines.h"
#include "include/types.h"
#include "src/gato_utils.cuh"
#include "src/gato_schur.cuh"
#include "src/gato_pcg.cuh"

#include <sstream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

namespace py = pybind11;
int gato_linsys(int *d_G_row, int *d_G_col, float *d_G_val,
                int *d_C_row, int *d_C_col, float *d_C_val,
                float *d_g_val,
                float *d_c_val,
                float *lambda, float *dz,
                bool warm_start, float eps, int max_iter){

    float *d_S, *d_Pinv, *d_gamma, *d_lambda, *d_dz;
    int pcg_iters;
    float rho = .001;

    float *d_G_dense, *d_C_dense;
    cuda_calloc((void **)&d_G_dense, KKT_G_DENSE_SIZE_BYTES);
    cuda_calloc((void **)&d_C_dense, KKT_C_DENSE_SIZE_BYTES);


    cuda_malloc((void **)&d_S,     3*STATES_SQ*KNOT_POINTS*sizeof(float));
    cuda_malloc((void **)&d_Pinv,  3*STATES_SQ*KNOT_POINTS*sizeof(float));
    cuda_malloc((void **)&d_gamma, STATE_SIZE*KNOT_POINTS*sizeof(float));
    cuda_malloc((void **)&d_dz, ((STATES_S_CONTROLS)*KNOT_POINTS-CONTROL_SIZE)*sizeof(float));

    cuda_calloc((void **)&d_lambda, STATE_SIZE*KNOT_POINTS*sizeof(float));
    if(warm_start)
        gpuErrchk( cudaMemcpy(d_lambda, lambda, STATE_SIZE*KNOT_POINTS*sizeof(float), cudaMemcpyHostToDevice));

#if DEBUG_MODE
        cudaDeviceSynchronize();
#endif

    form_schur(d_G_row, d_G_col, d_G_val, d_G_dense,
               d_C_row, d_C_col, d_C_val, d_C_dense,
               d_g_val,
               d_c_val,
               d_S, d_Pinv, d_gamma, rho);
    
    pcg_iters = solve_pcg<float>(d_S, d_Pinv, d_gamma, d_lambda, warm_start, eps, max_iter);

#if DEBUG_MODE
        cudaDeviceSynchronize();
#endif

    compute_dz(d_G_dense, d_C_dense, d_g_val, d_lambda, d_dz);

#if DEBUG_MODE
    cudaDeviceSynchronize();
#endif

    gpuErrchk(cudaMemcpy(lambda, d_lambda, STATE_SIZE*KNOT_POINTS*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(dz, d_dz, (STATES_S_CONTROLS*KNOT_POINTS-CONTROL_SIZE)*sizeof(float), cudaMemcpyDeviceToHost));

    cuda_free((void **)d_G_dense);
    cuda_free((void **)d_C_dense);

    gpuErrchk(cudaFree(d_S));
    gpuErrchk(cudaFree(d_Pinv));
    gpuErrchk(cudaFree(d_gamma));
    gpuErrchk(cudaFree(d_lambda));
    gpuErrchk(cudaFree(d_dz));
    return pcg_iters;
}

py::tuple main_call(std::vector<int> sG_indptr_vector, std::vector<int> sG_indices_vector, std::vector<float> sG_data_vector, 
                std::vector<int> sC_indptr_vector, std::vector<int> sC_indices_vector, std::vector<float> sC_data_vector, 
                std::vector<float> g_vector, std::vector<float> c_vector, std::vector<float> input_lambda, int testiters, float exit_tol, int max_iters, bool warm_start){
    
    
    int* G_row = sG_indptr_vector.data();
    int G_row_size_bytes = sG_indptr_vector.size() * sizeof(int);
    
    int* G_col = sG_indices_vector.data();
    int G_col_size_bytes = sG_indices_vector.size() * sizeof(int);

    float* G_val = sG_data_vector.data();
    int G_val_size_bytes = sG_data_vector.size() * sizeof(float);
    
    int* C_row = sC_indptr_vector.data();
    int C_row_size_bytes = sC_indptr_vector.size() * sizeof(int);
    
    int* C_col = sC_indices_vector.data();
    int C_col_size_bytes = sC_indices_vector.size() * sizeof(int);

    float* C_val = sC_data_vector.data();
    int C_val_size_bytes = sC_data_vector.size() * sizeof(float);
    
    float* g_val = g_vector.data();
    int g_size_bytes = g_vector.size() * sizeof(float);
    
    float* c_val = c_vector.data();
    int c_size_bytes = c_vector.size() * sizeof(float);

#if DEBUG_MODE
/*
    std::cout << "G row \n";
    for(unsigned i = 0; i < G_row_size_bytes/sizeof(int); i++){
        std::cout << G_row[i] << " ";
    }
    std::cout << "\nG col \n";
    for(unsigned i = 0; i < G_col_size_bytes/sizeof(int); i++){
        std::cout << G_col[i] << " ";
    }
    std::cout << "\nG val \n";
    for(unsigned i = 0; i < G_val_size_bytes/sizeof(int); i++){
        std::cout << G_val[i] << " ";
    }
*/
#endif  /* #if DEBUG_MODE */

    float dz[(STATES_S_CONTROLS)*KNOT_POINTS-CONTROL_SIZE];


    float *d_G_val, *d_C_val, *d_g_val, *d_c_val;
    int *d_G_row, *d_G_col, *d_C_row, *d_C_col;

    cuda_malloc((void **)&d_G_val, G_val_size_bytes);
    cuda_malloc((void **)&d_G_row, G_row_size_bytes);
    cuda_malloc((void **)&d_G_col, G_col_size_bytes);
    cuda_malloc((void **)&d_C_val, C_val_size_bytes);
    cuda_malloc((void **)&d_C_row, C_row_size_bytes);
    cuda_malloc((void **)&d_C_col, C_col_size_bytes);
    cuda_malloc((void **)&d_g_val, g_size_bytes);
    cuda_malloc((void **)&d_c_val, c_size_bytes);
    
    /// TODO: wrap cudamemcpy like malloc
    gpuErrchk( cudaMemcpy(d_G_val, G_val, G_val_size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_G_row, G_row, G_row_size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_G_col, G_col, G_col_size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_C_val, C_val, C_val_size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_C_row, C_row, C_row_size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_C_col, C_col, C_col_size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_g_val, g_val, g_size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_c_val, c_val, c_size_bytes, cudaMemcpyHostToDevice));



    float lambda[STATE_SIZE*KNOT_POINTS];
    for(int i=0;i<STATE_SIZE*KNOT_POINTS;i++){
        lambda[i] = input_lambda[i];
    }

    float times[testiters];
    cudaEvent_t start, stop;    

    for(int i = 0; i < testiters; i++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        int iters = gato_linsys(d_G_row, d_G_col, d_G_val,
                                d_C_row, d_C_col, d_C_val,
                                d_g_val,
                                d_c_val,
                                lambda, dz,
                                warm_start, exit_tol, max_iters);

        cudaEventRecord(stop);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        
        float e1;
        cudaEventElapsedTime(&e1, start, stop);
        times[i] = e1;

        if(i==0){
            printf("first run PCG terminated in %d iterations, time:  %f\n", iters, e1);
        }
    }

    for(int i = 1; i < testiters; i++){
        times[0] += times[i];
    }

    printf("avg time: %f\n", times[0]/testiters);

    
    // for(int i =0; i < STATE_SIZE*KNOT_POINTS; i++)
    //      printf("%f\n", lambda[i]);
    // printf("\n\ndz\n");
    // for(int i =0; i < (STATES_S_CONTROLS)*KNOT_POINTS-CONTROL_SIZE; i++)
    //     printf("%f\n", dz[i]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    gpuErrchk( cudaFree(d_G_val));
    gpuErrchk( cudaFree(d_G_row));
    gpuErrchk( cudaFree(d_G_col));
    gpuErrchk( cudaFree(d_C_val));
    gpuErrchk( cudaFree(d_C_row));
    gpuErrchk( cudaFree(d_C_col));
    gpuErrchk( cudaFree(d_g_val));
    gpuErrchk( cudaFree(d_c_val));
    
    
	py::list p_lambda;
    for(int i=0;i<=STATE_SIZE*KNOT_POINTS;i++){
        p_lambda.append(lambda[i]);
    }
    py::list p_dz;
    for(int i=0;i<=(STATES_S_CONTROLS)*KNOT_POINTS-CONTROL_SIZE;i++){
        p_dz.append(dz[i]);
    }

    py::tuple ans = py::make_tuple(p_lambda, p_dz);
    return ans;

    
}

PYBIND11_MODULE(gpu_library, m)
{
  m.def("linsys_solve", &main_call, py::return_value_policy::move);
}
