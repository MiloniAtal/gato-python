#ifndef GATO_PCG_CUH
#define GATO_PCG_CUH


#include "../include/types.h"
#include "../include/gato_defines.h"

#include "../cuda_utils/cuda_malloc.cuh"
#include "gato_utils.cuh"



/*******************************************************************************
*                        private functions pcg solve                           *
*******************************************************************************/

template<typename T, bool USE_TRACE = false>
__device__
void parallelPCG_inner_fixed(float  *d_S, float  *d_pinv, float  *d_gamma,  				// block-local constant temporary variable inputs
                        float  *d_lambda, float  *d_r, float  *d_p, float  *d_v, float  *d_eta_new, float  *d_r_tilde, float  *d_upsilon,	// global vectors and scalars
                        float  *s_temp, int *iters, int maxIters, T exitTol, 	    // shared mem for use in CG step and constants
                        cgrps::thread_block block, cgrps::grid_group grid){
                            
    //Initialise shared memory
    float  *s_lambda = s_temp;
    float  *s_r_tilde = s_lambda + STATE_SIZE;
    float  *s_upsilon = s_r_tilde + STATE_SIZE;
    float  *s_v_b = s_upsilon + STATE_SIZE;
    float  *s_eta_new_b = s_v_b + STATE_SIZE;

    float  *s_r = s_eta_new_b + STATE_SIZE;
    float  *s_p = s_r + 3*STATE_SIZE;

    float  *s_r_b = s_r + STATE_SIZE;
    float  *s_p_b = s_p + STATE_SIZE;

    T alpha, beta;
    T eta = static_cast<T>(0);	T eta_new = static_cast<T>(0);

    // Need to initialise *s_S, float  *s_pinv, float  *s_gamma and input all required as dynamic memory
    float  *s_S = s_p + 3 * STATE_SIZE;
    float  *s_pinv = s_S + 3*STATE_SIZE*STATE_SIZE;
    // float  *s_gamma = s_pinv + 3*STATE_SIZE*STATE_SIZE;

    // Used when writing to device memory
    int bIndStateSize;

    if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
        *iters = maxIters;
    }

    // Initililiasation before the main-pcg loop
    // note that in this formulation we always reset lambda to 0 and therefore we can simplify this step
    // Therefore, s_r_b = s_gamma_b

    for( unsigned block_number = GATO_BLOCK_ID; block_number < KNOT_POINTS; block_number += GATO_NUM_BLOCKS){

        // directly write to device memory
        bIndStateSize = STATE_SIZE * block_number;
        // We find the s_r, load it into device memory, initialise lambda to 0
        for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            d_r[bIndStateSize + ind] = d_gamma[STATE_SIZE * block_number + ind]; 
            d_lambda[bIndStateSize + ind] = static_cast<T>(0);
        }

        block.sync();
    }

    if(GATO_LEAD_THREAD && GATO_LEAD_BLOCK){
        *d_eta_new = static_cast<T>(0);

        //TODO:remove redundant
        *d_v = static_cast<T>(0);
    }
    // Need to sync before loading from other blocks
    grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
    // if(GATO_LEAD_THREAD && GATO_LEAD_BLOCK){
    //         print_raw_shared_vector<float ,STATE_SIZE*KNOT_POINTS>(d_r);
    // }
    // grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
    
    
    for( unsigned block_number = GATO_BLOCK_ID; block_number < KNOT_POINTS; block_number += GATO_NUM_BLOCKS){
        // load s_r_b, pinv
        bIndStateSize = STATE_SIZE * block_number;
        for (unsigned ind= GATO_THREAD_ID; ind < 3*STATE_SIZE*STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            s_pinv[ind] = d_pinv[bIndStateSize*STATE_SIZE*3 + ind]; 
        }
        loadBlockTriDiagonal_offDiagonal_fixed<float ,STATE_SIZE, KNOT_POINTS>(s_r,&d_r[bIndStateSize],block_number,block,grid);
        block.sync();
        matVecMultBlockTriDiagonal_fixed<float ,STATE_SIZE, KNOT_POINTS>(s_r_tilde,s_pinv,s_r,block_number,block,grid);
        block.sync();

        dotProd<float ,STATE_SIZE>(s_eta_new_b,s_r_b,s_r_tilde,block);
        block.sync();

        // We copy p from r_tilde and write to device, since it will be required by other blocks
        //write back s_r_tilde, s_p
        for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            d_p[bIndStateSize + ind] = s_r_tilde[ind];
            d_r_tilde[bIndStateSize + ind] = s_r_tilde[ind];
        }
        if(GATO_LEAD_THREAD){
            // printf("Partial sums of Block %d and Block Number %d: %f\n", GATO_BLOCK_ID, block_number,s_eta_new_b[0] );
            atomicAdd(d_eta_new,s_eta_new_b[0]);
        }
        block.sync();
    
        
    }
    grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
    eta = *d_eta_new;
    block.sync();
    // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
    //     printf("Before main loop eta %f > exitTol %f\n",eta, exitTol);
    // }
    for(unsigned iter = 0; iter < maxIters; iter++){
        if(GATO_LEAD_THREAD && GATO_LEAD_BLOCK){
            *d_v = static_cast<T>(0);
        }
        grid.sync();
        // for over rows, 
        for( unsigned block_number = GATO_BLOCK_ID; block_number < KNOT_POINTS; block_number += GATO_NUM_BLOCKS){

            bIndStateSize = STATE_SIZE * block_number;
            // s_S, s_p (already) load from device for that particular row
            for (unsigned ind = GATO_THREAD_ID; ind < 3*STATE_SIZE*STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                s_S[ind] = d_S[bIndStateSize * STATE_SIZE * 3 + ind]; 
            }
            block.sync();


            loadBlockTriDiagonal_offDiagonal_fixed<float ,STATE_SIZE,KNOT_POINTS>(s_p,&d_p[bIndStateSize],block_number,block,grid);
            block.sync();
            matVecMultBlockTriDiagonal_fixed<float ,STATE_SIZE,KNOT_POINTS>(s_upsilon,s_S,s_p,block_number,block,grid);
            block.sync();
            dotProd<float ,STATE_SIZE>(s_v_b,s_p_b,s_upsilon,block);
            block.sync();

            // only upsilon needs to be written to device memory
            for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                d_upsilon[bIndStateSize + ind] = s_upsilon[ind];
            }

            if(GATO_LEAD_THREAD){
                atomicAdd(d_v,s_v_b[0]);
            }
            block.sync();
            
        }
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        for( unsigned block_number = GATO_BLOCK_ID; block_number < KNOT_POINTS; block_number += GATO_NUM_BLOCKS){

            bIndStateSize = STATE_SIZE * block_number;
            
            // load s_p, s_lambda, s_upsilon, s_r
            for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                s_p_b[ind] = d_p[bIndStateSize + ind];
                s_lambda[ind] = d_lambda[bIndStateSize + ind];
                s_upsilon[ind] = d_upsilon[bIndStateSize + ind];
                s_r_b[ind] = d_r[bIndStateSize + ind];
            }

            alpha = eta / *d_v;

            //Dont need this
            block.sync();

            // Move this loop into a function, write  back lambda and r
            for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                s_lambda[ind] += alpha * s_p_b[ind];
                s_r_b[ind] -= alpha * s_upsilon[ind];
                d_lambda[bIndStateSize + ind] = s_lambda[ind];
                d_r[bIndStateSize + ind] = s_r_b[ind];
            }
            block.sync();
        }

        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        
        if(GATO_LEAD_THREAD && GATO_LEAD_BLOCK){
            *d_eta_new = static_cast<T>(0);
        }
        block.sync();
        for( unsigned block_number = GATO_BLOCK_ID; block_number < KNOT_POINTS; block_number += GATO_NUM_BLOCKS){

            bIndStateSize = STATE_SIZE * block_number;
            // load s_r (already), s_pinv
            for (unsigned ind = GATO_THREAD_ID; ind < 3*STATE_SIZE*STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                s_pinv[ind] = d_pinv[bIndStateSize * STATE_SIZE * 3 + ind]; 
            }

            loadBlockTriDiagonal_offDiagonal_fixed<float ,STATE_SIZE,KNOT_POINTS>(s_r,&d_r[bIndStateSize],block_number,block,grid);
            block.sync();
            matVecMultBlockTriDiagonal_fixed<float ,STATE_SIZE,KNOT_POINTS>(s_r_tilde,s_pinv,s_r,block_number,block,grid);
            block.sync();
            dotProd<float ,STATE_SIZE>(s_eta_new_b,s_r_tilde,s_r_b,block);
            block.sync();
            // write back r_tilde
            for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                d_r_tilde[bIndStateSize + ind] = s_r_tilde[ind];
            }

            if(GATO_LEAD_THREAD){
                atomicAdd(d_eta_new,s_eta_new_b[0]);
            }
            block.sync();
        }
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        
        // test for exit
        eta_new = *d_eta_new;
        
        if(abs(eta_new) < exitTol){
            // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
            //     printf("Breaking at iter %d with eta %f > exitTol %f------------------------------------------------------\n", iter, abs(eta_new), exitTol);
            // }
            if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
                *iters = iter;
            }
            
            break;
        }
        
        // else compute d_p for next loop
        else{

            beta = eta_new / eta;
            for( unsigned block_number = GATO_BLOCK_ID; block_number < KNOT_POINTS; block_number += GATO_NUM_BLOCKS){

                bIndStateSize = STATE_SIZE * block_number;
                // load s_p, s_r_tilde, write back s_p
                for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                    s_p_b[ind] = d_p[bIndStateSize + ind];
                    s_r_tilde[ind] = d_r_tilde[bIndStateSize + ind];
                    s_p_b[ind] = s_r_tilde[ind] + beta*s_p_b[ind];
                    d_p[bIndStateSize + ind] = s_p_b[ind];
                }  
            
            }
            eta = eta_new;
            block.sync();
        }
        // then global sync for next loop
        // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
        //     printf("Executing iter %d with eta %f > exitTol %f------------------------------------------------------\n", iter, abs(eta_new), exitTol);
        // }
        // then global sync for next loop
        
        grid.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
        
    } 
    
}

template<typename T, bool USE_TRACE = false>
__global__
void parallelPCG_fixed(float  *d_S, float  *d_pinv, float  *d_gamma,  				// block-local constant temporary variable inputs
                        float  *d_lambda, float  *d_r, float  *d_p, float  *d_v, float  *d_eta_new, float  *d_r_tilde, float  *d_upsilon,	// global vectors and scalars
                        int *iters, int maxIters=100, float exitTol=1e-6			    // shared mem for use in CG step and constants 
                        ){

    __shared__ T s_temp[3*STATE_SIZE*STATE_SIZE + 3*STATE_SIZE*STATE_SIZE + STATE_SIZE + 11 * STATE_SIZE];

    cgrps::thread_block block = cgrps::this_thread_block();	 
    cgrps::grid_group grid = cgrps::this_grid();
    
    grid.sync();
    parallelPCG_inner_fixed<float, false>(d_S, d_pinv, d_gamma, d_lambda, d_r, d_p, d_v, d_eta_new, d_r_tilde, d_upsilon, s_temp, iters, maxIters, exitTol, block, grid);
    grid.sync();
}

template <typename T, bool USE_TRACE = false>
__device__
void parallelPCG_inner(float  *s_S, float  *s_pinv, float  *s_gamma,  				// block-local constant temporary variable inputs
                        float  *d_lambda, float  *d_r, float  *d_p, float  *d_v, float  *d_eta_new,	// global vectors and scalars
                        float  *s_temp, int *iters,	int maxIters, float exitTol, 		    // shared mem for use in CG step and constants
                        cgrps::thread_block block, cgrps::grid_group grid){                      
    //Initialise shared memory
    float  *s_lambda = s_temp;
    float  *s_r_tilde = s_lambda + STATE_SIZE;
    float  *s_upsilon = s_r_tilde + STATE_SIZE;
    float  *s_v_b = s_upsilon + STATE_SIZE;
    float  *s_eta_new_b = s_v_b + STATE_SIZE;

    float  *s_r = s_eta_new_b + STATE_SIZE;
    float  *s_p = s_r + 3*STATE_SIZE;

    float  *s_r_b = s_r + STATE_SIZE;
    float  *s_p_b = s_p + STATE_SIZE;

    T alpha, beta;
    T eta = static_cast<T>(0);	T eta_new = static_cast<T>(0);

    // Used when writing to device memory
    int bIndStateSize = STATE_SIZE * GATO_BLOCK_ID;

    // Initililiasation before the main-pcg loop
    // note that in this formulation we always reset lambda to 0 and therefore we can simplify this step
    // Therefore, s_r_b = s_gamma_b

    // We find the s_r, load it into device memory, initialise lambda to 0
    for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
        s_r_b[ind] = s_gamma[ind];
        d_r[bIndStateSize + ind] = s_r_b[ind]; 
        s_lambda[ind] = static_cast<T>(0);
    }
    // Make eta_new zero
    if(GATO_LEAD_THREAD && GATO_LEAD_BLOCK){
        *d_eta_new = static_cast<T>(0);
        *d_v = static_cast<T>(0);
    }

    if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
        *iters = maxIters;
    }
    // Need to sync before loading from other blocks
    grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
    loadBlockTriDiagonal_offDiagonal<float ,STATE_SIZE>(s_r,&d_r[bIndStateSize],block,grid);
    block.sync();
    matVecMultBlockTriDiagonal<float ,STATE_SIZE>(s_r_tilde,s_pinv,s_r,block,grid);
    block.sync();

    // We copy p from r_tilde and write to device, since it will be required by other blocks
    for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
        s_p_b[ind] = s_r_tilde[ind];
        d_p[bIndStateSize + ind] = s_p_b[ind]; 
    }

    dotProd<float ,STATE_SIZE>(s_eta_new_b,s_r_b,s_r_tilde,block);
    block.sync();

    if(GATO_LEAD_THREAD){
        // printf("Partial sums of Block %d: %f\n", GATO_BLOCK_ID, s_eta_new_b[0] );
        atomicAdd(d_eta_new,s_eta_new_b[0]);
    }

    grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
    eta = *d_eta_new;
    block.sync();

    // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
    //     printf("Before main loop eta %f > exitTol %f\n",eta, exitTol);
    // }
    
    for(unsigned iter = 0; iter < maxIters; iter++){
        loadBlockTriDiagonal_offDiagonal<float ,STATE_SIZE>(s_p,&d_p[bIndStateSize],block,grid);
        block.sync();
        matVecMultBlockTriDiagonal<float ,STATE_SIZE>(s_upsilon,s_S,s_p,block,grid);
        block.sync();
        dotProd<float ,STATE_SIZE>(s_v_b,s_p_b,s_upsilon,block);
        block.sync();

        if(GATO_LEAD_THREAD){
            atomicAdd(d_v,s_v_b[0]);
            // Ideally move to just before calculation but then needs extra sync
            if(GATO_LEAD_BLOCK){
                *d_eta_new = static_cast<T>(0);
            }
        }
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        alpha = eta / *d_v;

        block.sync();
        if(false){
            printf("d_pSp[%f] -> alpha[%f]\n",*d_v,alpha);
        }

        block.sync();

        // Move this loop into a function
        for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            s_lambda[ind] += alpha * s_p_b[ind];
            s_r_b[ind] -= alpha * s_upsilon[ind];
            d_r[bIndStateSize + ind] = s_r_b[ind];
            }
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER

        loadBlockTriDiagonal_offDiagonal<float ,STATE_SIZE>(s_r,&d_r[bIndStateSize],block,grid);
        block.sync();
        matVecMultBlockTriDiagonal<float ,STATE_SIZE>(s_r_tilde,s_pinv,s_r,block,grid);
        block.sync();
        dotProd<float ,STATE_SIZE>(s_eta_new_b,s_r_b,s_r_tilde,block);
        block.sync();
        if(GATO_LEAD_THREAD){
            atomicAdd(d_eta_new,s_eta_new_b[0]);
            // Ideally move to just before calculation but then needs extra sync
            if(GATO_LEAD_BLOCK){
                *d_v = static_cast<T>(0);
            }
        }
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        eta_new = *d_eta_new;

        block.sync();
#if DEBUG_MODE
        if(GATO_BLOCK_ID==0&&GATO_THREAD_ID==0){
            printf("eta_new[%f]\n",abs(eta_new));
        }
#endif /* #if DEBUG_MODE */
        block.sync();

        if(abs(eta_new) < exitTol){

            if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
                *iters = iter;
            }

            break;
        }
        
        // else compute d_p for next loop
        else{
            beta = eta_new / eta;
            for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                s_p_b[ind] = s_r_tilde[ind] + beta*s_p_b[ind];
                d_p[bIndStateSize + ind] = s_p_b[ind];
            }
            eta = eta_new;
        }

        // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
        //     printf("Executing iter %d with eta %f > exitTol %f------------------------------------------------------\n", iter, abs(eta_new), exitTol);
        // }
        // then global sync for next loop
        grid.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
        
    }
    // save final lambda to global
    block.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
    for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
        d_lambda[bIndStateSize + ind] = s_lambda[ind];
    }
    
    grid.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
    
}

template <typename T, bool USE_TRACE = false>
__global__
void parallelPCG(float  *d_S, float  *d_pinv, float  *d_gamma,  				// block-local constant temporary variable inputs
                        float  *d_lambda, float  *d_r, float  *d_p, float  *d_v, float  *d_eta_new,	// global vectors and scalars
                        int *iters, int maxIters=100, T exitTol = 1e-6 	// shared mem for use in CG step and constants
                        ){

    __shared__ T s_temp[3*STATE_SIZE*STATE_SIZE + 3*STATE_SIZE*STATE_SIZE + STATE_SIZE + 11 * STATE_SIZE];
    float  *s_S = s_temp;
    float  *s_pinv = s_S +3*STATE_SIZE*STATE_SIZE;
    float  *s_gamma = s_pinv + 3*STATE_SIZE*STATE_SIZE;
    float  *shared_mem = s_gamma + STATE_SIZE;

    cgrps::thread_block block = cgrps::this_thread_block();	 
    cgrps::grid_group grid = cgrps::this_grid();

    int bIndStateSize = STATE_SIZE * GATO_BLOCK_ID;
    for (unsigned ind = GATO_THREAD_ID; ind < 3 * STATE_SIZE * STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
        s_S[ind] = d_S[bIndStateSize*STATE_SIZE*3 + ind];
        s_pinv[ind] = d_pinv[bIndStateSize*STATE_SIZE*3 + ind];
    }
    for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
        s_gamma[ind] = d_gamma[bIndStateSize + ind];
    }
    grid.sync();

    parallelPCG_inner<float>(s_S, s_pinv, s_gamma, d_lambda, d_r, d_p, d_v, d_eta_new, shared_mem, iters, maxIters, exitTol, block, grid);
    grid.sync();
}


template <typename T, bool USE_TRACE = false>
__device__
void parallelCG_inner(float  *s_S, float  *s_gamma,  				// block-local constant temporary variable inputs
                        float  *d_lambda, float  *d_r, float  *d_p, float  *d_v, float  *d_eta_new,	// global vectors and scalars
                        float  *s_temp, T exitTol, unsigned maxIters,			    // shared mem for use in CG step and constants
                        cgrps::thread_block block, cgrps::grid_group grid){                      
    //Initialise shared memory
    float  *s_lambda = s_temp;
    float  *s_upsilon = s_lambda+ STATE_SIZE;
    float  *s_v_b = s_upsilon + STATE_SIZE;
    float  *s_eta_new_b = s_v_b + STATE_SIZE;

    float  *s_r = s_eta_new_b + STATE_SIZE;
    float  *s_p = s_r + 3*STATE_SIZE;

    float  *s_r_b = s_r + STATE_SIZE;
    float  *s_p_b = s_p + STATE_SIZE;

    T alpha, beta;
    T eta = static_cast<T>(0);	T eta_new = static_cast<T>(0);

    // Used when writing to device memory
    int bIndStateSize = STATE_SIZE * GATO_BLOCK_ID;

    // Initililiasation before the main-pcg loop
    // note that in this formulation we always reset lambda to 0 and therefore we can simplify this step
    // Therefore, s_r_b = s_gamma_b

    // We find the s_r, load it into device memory, initialise lambda to 0
    for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
        s_r_b[ind] = s_gamma[ind];
        d_r[bIndStateSize + ind] = s_r_b[ind]; 
        s_lambda[ind] = static_cast<T>(0);
    }
    // Make eta_new zero
    if(GATO_LEAD_THREAD && GATO_LEAD_BLOCK){
        *d_eta_new = static_cast<T>(0);
        *d_v = static_cast<T>(0);
    }

    // Need to sync before loading from other blocks
    grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER


    // We copy p from r_tilde and write to device, since it will be required by other blocks
    for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
        s_p_b[ind] = s_r_b[ind];
        d_p[bIndStateSize + ind] = s_p_b[ind]; 
    }

    dotProd<float ,STATE_SIZE>(s_eta_new_b,s_r_b,s_r_b,block);
    block.sync();

    if(GATO_LEAD_THREAD){
        // printf("Partial sums of Block %d: %f\n", GATO_BLOCK_ID, s_eta_new_b[0] );
        atomicAdd(d_eta_new,s_eta_new_b[0]);
    }

    grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
    eta = *d_eta_new;
    block.sync();

    // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
    //     printf("Before main loop eta %f > exitTol %f\n",eta, exitTol);
    // }
    
    for(unsigned iter = 0; iter < maxIters; iter++){
        loadBlockTriDiagonal_offDiagonal<float ,STATE_SIZE>(s_p,&d_p[bIndStateSize],block,grid);
        block.sync();
        matVecMultBlockTriDiagonal<float ,STATE_SIZE>(s_upsilon,s_S,s_p,block,grid);
        block.sync();
        dotProd<float ,STATE_SIZE>(s_v_b,s_p_b,s_upsilon,block);
        block.sync();

        if(GATO_LEAD_THREAD){
            atomicAdd(d_v,s_v_b[0]);
            // Ideally move to just before calculation but then needs extra sync
            if(GATO_LEAD_BLOCK){
                *d_eta_new = static_cast<T>(0);
            }
        }
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        alpha = eta / *d_v;

        block.sync();
        if(false){
            printf("d_pSp[%f] -> alpha[%f]\n",*d_v,alpha);
        }

        block.sync();

        // Move this loop into a function
        for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            s_lambda[ind] += alpha * s_p_b[ind];
            s_r_b[ind] -= alpha * s_upsilon[ind];
            d_r[bIndStateSize + ind] = s_r_b[ind];
            }
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER


        dotProd<float ,STATE_SIZE>(s_eta_new_b,s_r_b,s_r_b,block);
        block.sync();
        if(GATO_LEAD_THREAD){
            atomicAdd(d_eta_new,s_eta_new_b[0]);
            // Ideally move to just before calculation but then needs extra sync
            if(GATO_LEAD_BLOCK){
                *d_v = static_cast<T>(0);
            }
        }
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        eta_new = *d_eta_new;

        block.sync();
        if(false){
            printf("eta_new[%f]\n",eta_new);
        }
        block.sync();

        if(abs(eta_new) < exitTol){
            break;
        }
        
        // else compute d_p for next loop
        else{
            beta = eta_new / eta;
            for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                s_p_b[ind] = s_r_b[ind] + beta*s_p_b[ind];
                d_p[bIndStateSize + ind] = s_p_b[ind];
            }
            eta = eta_new;
        }

        // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
        //     printf("Executing iter %d with eta %f > exitTol %f------------------------------------------------------\n", iter, abs(eta_new), exitTol);
        // }
        // then global sync for next loop
        grid.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
        
    }
    // save final lambda to global
    block.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
    for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
        d_lambda[bIndStateSize + ind] = s_lambda[ind];
    }
    
    grid.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
    
}

template <typename T, bool USE_TRACE = false>
__global__
void parallelCG(float  *d_S, float  *d_pinv, float  *d_gamma,  				// block-local constant temporary variable inputs
                        float  *d_lambda, float  *d_r, float  *d_p, float  *d_v, float  *d_eta_new,	// global vectors and scalars
                        T exitTol = 1e-6, unsigned maxIters=100			    // shared mem for use in CG step and constants
                        ){

    __shared__ T s_temp[3*STATE_SIZE*STATE_SIZE + 3*STATE_SIZE*STATE_SIZE + STATE_SIZE + 10 * STATE_SIZE];
    float  *s_S = s_temp;
    float  *s_pinv = s_S +3*STATE_SIZE*STATE_SIZE;
    float  *s_gamma = s_pinv + 3*STATE_SIZE*STATE_SIZE;
    float  *shared_mem = s_gamma + STATE_SIZE;

    cgrps::thread_block block = cgrps::this_thread_block();	 
    cgrps::grid_group grid = cgrps::this_grid();

    int bIndStateSize = STATE_SIZE * GATO_BLOCK_ID;
    for (unsigned ind = GATO_THREAD_ID; ind < 3 * STATE_SIZE * STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
        s_S[ind] = d_S[bIndStateSize*STATE_SIZE*3 + ind];
        s_pinv[ind] = d_pinv[bIndStateSize*STATE_SIZE*3 + ind];
    }
    for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
        s_gamma[ind] = d_gamma[bIndStateSize + ind];
    }
    grid.sync();
    //Fix maxiter and exitTol issue
    parallelCG_inner<float, false>(s_S, s_gamma, d_lambda, d_r, d_p, d_v, d_eta_new, shared_mem, 1e-4, 100, block, grid);
    grid.sync();
}

/*******************************************************************************
*                                   API                                        *
*******************************************************************************/
    

template <typename T>
int solve_pcg(float  *d_S, float  *d_Pinv, float  *d_gamma, float *d_lambda, bool warm_start, float  eps, int max_iter){


    float  *d_r, *d_p, *d_v, *d_eta_new, *d_r_tilde, *d_upsilon;
    unsigned *iters;
    

    unsigned sharedMemSize;

    cuda_malloc((void **)&d_r, STATE_SIZE*KNOT_POINTS*sizeof(T));
    cuda_malloc((void **)&d_p, STATE_SIZE*KNOT_POINTS*sizeof(T));
    cuda_malloc((void **)&d_v, sizeof(T));
    cuda_malloc((void **)&d_eta_new, sizeof(T));
    cuda_malloc((void **)&d_r_tilde, STATE_SIZE*KNOT_POINTS*sizeof(T));
    cuda_malloc((void **)&d_upsilon, STATE_SIZE*KNOT_POINTS*sizeof(T));
    cuda_malloc((void **)&iters, sizeof(int));

    
    sharedMemSize = (2 * 3 * STATES_SQ + STATE_SIZE + 11 * STATE_SIZE)*sizeof(T);

    dim3 grid(KNOT_POINTS,1,1);
    dim3 block(STATE_SIZE,1,1);
    
    void *my_kernel = (void *)parallelPCG<float , false>;
    int num_blocks = check_sms<float>(my_kernel, block);

    //Each block does exactly one row
    if(false){
        void *kernelArgsCG[] = {
            (void *)&d_S,
            (void *)&d_Pinv,
            (void *)&d_gamma, 
            (void *)&d_lambda,
            (void *)&d_r,
            (void *)&d_p,
            (void *)&d_v,
            (void *)&d_eta_new,
            (void *)&iters,
            (void *)&max_iter,
            (void *)&eps
        };
        // printf("Using the old algo \n");
        void *cg = (void *)parallelCG<float>;
        sharedMemSize = (1*3 * STATES_SQ + STATE_SIZE + 10 * STATE_SIZE)*sizeof(T);
        cudaLaunchCooperativeKernel(cg, grid, block, kernelArgsCG, sharedMemSize);
        gpuErrchk( cudaPeekAtLastError() );
    }
    else if(num_blocks == KNOT_POINTS){
        void *kernelArgs[] = {
            (void *)&d_S,
            (void *)&d_Pinv,
            (void *)&d_gamma, 
            (void *)&d_lambda,
            (void *)&d_r,
            (void *)&d_p,
            (void *)&d_v,
            (void *)&d_eta_new,
            (void *)&iters,
            (void *)&max_iter,
            (void *)&eps
        };
        // printf("Using the old algo \n");
        gpuErrchk(cudaLaunchCooperativeKernel(my_kernel, grid, block, kernelArgs, sharedMemSize));
        gpuErrchk( cudaPeekAtLastError() );
    }
    //Each blocks needs to do more rows
    else if(num_blocks < KNOT_POINTS){
        // printf("Using the new algo \n");
        void *kernelArgsFixed[] = {
            (void *)&d_S,
            (void *)&d_Pinv,
            (void *)&d_gamma, 
            (void *)&d_lambda,
            (void *)&d_r,
            (void *)&d_p,
            (void *)&d_v,
            (void *)&d_eta_new,
            (void *)&d_r_tilde,
            (void *)&d_upsilon,
            (void *)&iters,
            (void *)&max_iter,
            (void *)&eps
        };
        dim3 grid_fixed(num_blocks,1,1);
        void *my_kernel_fixed = (void *)parallelPCG_fixed<float , KNOT_POINTS>;
        gpuErrchk(cudaLaunchCooperativeKernel(my_kernel_fixed, grid_fixed, block, kernelArgsFixed, sharedMemSize));
        gpuErrchk( cudaPeekAtLastError() );
    }


    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_v);
    cudaFree(d_eta_new);
    cudaFree(d_r_tilde);
    cudaFree(d_upsilon);

    int _iters;
    gpuErrchk(cudaMemcpy(&_iters, iters, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(iters);

    return _iters;
}


#endif /* GATO_PCG_CUH */