#include "../include/types.h"
#include "../include/gato_defines.h"

#include "../cuda_utils/cuda_malloc.cuh"
#include "gato_utils.cuh"



/*******************************************************************************
 *                           Private Functions                                 *
 *******************************************************************************/

__device__
void gato_form_schur_jacobi_inner(float *d_G, float *d_C, float *d_g, float *d_c, float *d_S, float *d_Pinv, float *d_gamma, float *s_temp, unsigned blockrow){
    
    
    //  SPACE ALLOCATION IN SHARED MEM
    //  | phi_k | theta_k | thetaInv_k | gamma_k | block-specific...
    //     s^2      s^2         s^2         s
    float *s_phi_k = s_temp; 	                            	    // phi_k        states^2
    float *s_theta_k = s_phi_k + STATES_SQ; 			            // theta_k      states^2
    float *s_thetaInv_k = s_theta_k + STATES_SQ; 			        // thetaInv_k   states^2
    float *s_gamma_k = s_thetaInv_k + STATES_SQ;                       // gamma_k      states
    float *s_end_main = s_gamma_k + STATE_SIZE;                               

    if(blockrow==0){

        //  LEADING BLOCK GOAL SHARED MEMORY STATE
        //  ...gamma_k | . | Q_N_I | q_N | . | Q_0_I | q_0 | scatch
        //              s^2   s^2     s   s^2   s^2     s      ? 
    
        float *s_QN = s_end_main;
        float *s_QN_i = s_QN + STATE_SIZE * STATE_SIZE;
        float *s_qN = s_QN_i + STATE_SIZE * STATE_SIZE;
        float *s_Q0 = s_qN + STATE_SIZE;
        float *s_Q0_i = s_Q0 + STATE_SIZE * STATE_SIZE;
        float *s_q0 = s_Q0_i + STATE_SIZE * STATE_SIZE;
        float *s_end = s_q0 + STATE_SIZE;

        // scratch space
        float *s_R_not_needed = s_end;
        float *s_r_not_needed = s_R_not_needed + CONTROL_SIZE * CONTROL_SIZE;
        float *s_extra_temp = s_r_not_needed + CONTROL_SIZE * CONTROL_SIZE;

        __syncthreads();//----------------------------------------------------------------

        gato_memcpy<float>(s_Q0, d_G, STATES_SQ);
        gato_memcpy<float>(s_QN, d_G+(KNOT_POINTS-1)*(STATES_SQ+CONTROLS_SQ), STATES_SQ);
        gato_memcpy<float>(s_q0, d_g, STATE_SIZE);
        gato_memcpy<float>(s_qN, d_g+(KNOT_POINTS-1)*(STATE_SIZE+CONTROL_SIZE), STATE_SIZE);

        __syncthreads();//----------------------------------------------------------------


        // if(PRINT_THREAD){
        //     printf("Q0\n");
        //     printMat<float,STATE_SIZE,STATE_SIZE>(s_Q0,STATE_SIZE);
        //     printf("q0\n");
        //     printMat<float,1,STATE_SIZE>(s_q0,1);
        //     printf("QN\n");
        //     printMat<float,STATE_SIZE,STATE_SIZE>(s_QN,STATE_SIZE);
        //     printf("qN\n");
        //     printMat<float,1,STATE_SIZE>(s_qN,1);
        //     printf("start error\n");
        //     printMat<float,1,STATE_SIZE>(s_integrator_error,1);
        //     printf("\n");
        // }
        __syncthreads();//----------------------------------------------------------------
        
        // SHARED MEMORY STATE
        // | Q_N | . | q_N | Q_0 | . | q_0 | scatch
        

        // save -Q_0 in PhiInv spot 00
        store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
            s_Q0,                       // src     
            d_Pinv,                   // dst         
            1,                          // col
            blockrow,                    // blockrow
            -1                          //  multiplier
        );
        __syncthreads();//----------------------------------------------------------------


        // invert Q_N, Q_0
        loadIdentity<float, STATE_SIZE,STATE_SIZE>(s_Q0_i, s_QN_i);
        __syncthreads();//----------------------------------------------------------------
        invertMatrix<float, STATE_SIZE,STATE_SIZE,STATE_SIZE>(s_Q0, s_QN, s_extra_temp);
        
        __syncthreads();//----------------------------------------------------------------


        // if(PRINT_THREAD){
        //     printf("Q0Inv\n");
        //     printMat<floatSTATE_SIZE,STATE_SIZE>(s_Q0_i,STATE_SIZE);
        //     printf("QNInv\n");
        //     printMat<floatSTATE_SIZE,STATE_SIZE>(s_QN_i,STATE_SIZE);
        //     printf("theta\n");
        //     printMat<floatSTATE_SIZE,STATE_SIZE>(s_theta_k,STATE_SIZE);
        //     printf("thetaInv\n");
        //     printMat<floatSTATE_SIZE,STATE_SIZE>(s_thetaInv_k,STATE_SIZE);
        //     printf("\n");
        // }
        __syncthreads();//----------------------------------------------------------------

        // SHARED MEMORY STATE
        // | . | Q_N_i | q_N | . | Q_0_i | q_0 | scatch
        

        // compute gamma
        mat_vec_prod<float, STATE_SIZE, STATE_SIZE>(
            s_Q0_i,                                    
            s_q0,                                       
            s_gamma_k 
        );
        __syncthreads();//----------------------------------------------------------------
        

        // save -Q0_i in spot 00 in S
        store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
            s_Q0_i,                         // src             
            d_S,                            // dst              
            1,                              // col   
            blockrow,                        // blockrow         
            -1                              //  multiplier   
        );
        __syncthreads();//----------------------------------------------------------------


        // compute Q0^{-1}q0
        mat_vec_prod<float, STATE_SIZE, STATE_SIZE>(
            s_Q0_i,
            s_q0,
            s_Q0
        );
        __syncthreads();//----------------------------------------------------------------


        // SHARED MEMORY STATE
        // | . | Q_N_i | q_N | Q0^{-1}q0 | Q_0_i | q_0 | scatch


        // save -Q0^{-1}q0 in spot 0 in gamma
        for(unsigned ind = GATO_THREAD_ID; ind < STATES_SQ; ind += GATO_THREADS_PER_BLOCK){
            d_gamma[ind] = -s_Q0[ind];
        }
        __syncthreads();//----------------------------------------------------------------

    }
    else{                       // blockrow!=LEAD_BLOCK


        const unsigned C_set_size = STATES_SQ+STATES_P_CONTROLS;
        const unsigned G_set_size = STATES_SQ+CONTROLS_SQ;

        //  NON-LEADING BLOCK GOAL SHARED MEMORY STATE
        //  ...gamma_k | A_k | B_k | . | Q_k_I | . | Q_k+1_I | . | R_k_I | q_k | q_k+1 | r_k | integrator_error | extra_temp
        //               s^2   s*c  s^2   s^2   s^2    s^2    s^2   s^2     s      s      s          s                <s^2?

        float *s_Ak = s_end_main; 								
        float *s_Bk = s_Ak +        STATES_SQ;
        float *s_Qk = s_Bk +        STATES_P_CONTROLS; 	
        float *s_Qk_i = s_Qk +      STATES_SQ;	
        float *s_Qkp1 = s_Qk_i +    STATES_SQ;
        float *s_Qkp1_i = s_Qkp1 +  STATES_SQ;
        float *s_Rk = s_Qkp1_i +    STATES_SQ;
        float *s_Rk_i = s_Rk +      CONTROLS_SQ;
        float *s_qk = s_Rk_i +      CONTROLS_SQ; 	
        float *s_qkp1 = s_qk +      STATE_SIZE; 			
        float *s_rk = s_qkp1 +      STATE_SIZE;
        float *s_end = s_rk +       CONTROL_SIZE;
        
        // scratch
        float *s_extra_temp = s_end;
        

        // if(PRINT_THREAD){
        //     printf("xk\n");
        //     printMat<float1,STATE_SIZE>(s_xux,1);
        //     printf("uk\n");
        //     printMat<float1,CONTROL_SIZE>(&s_xux[STATE_SIZE],1);
        //     printf("xkp1\n");
        //     printMat<float1,STATE_SIZE>(&s_xux[STATE_SIZE+CONTROL_SIZE],1);
        //     printf("\n");
        // }

        __syncthreads();//----------------------------------------------------------------

        gato_memcpy<float>(s_Ak,   d_C+      (blockrow-1)*C_set_size,                        STATES_SQ);
        gato_memcpy<float>(s_Bk,   d_C+      (blockrow-1)*C_set_size+STATES_SQ,              STATES_P_CONTROLS);
        gato_memcpy<float>(s_Qk,   d_G+      (blockrow-1)*G_set_size,                        STATES_SQ);
        gato_memcpy<float>(s_Qkp1, d_G+    (blockrow*G_set_size),                          STATES_SQ);
        gato_memcpy<float>(s_Rk,   d_G+      ((blockrow-1)*G_set_size+STATES_SQ),            CONTROLS_SQ);
        gato_memcpy<float>(s_qk,   d_g+      (blockrow-1)*(STATES_S_CONTROLS),               STATE_SIZE);
        gato_memcpy<float>(s_qkp1, d_g+    (blockrow)*(STATES_S_CONTROLS),                 STATE_SIZE);
        gato_memcpy<float>(s_rk,   d_g+      ((blockrow-1)*(STATES_S_CONTROLS)+STATE_SIZE),  CONTROL_SIZE);

        __syncthreads();//----------------------------------------------------------------

#if DEBUG_MODE    
        if(GATO_BLOCK_ID==1 && GATO_THREAD_ID==0){
            printf("Ak\n");
            printMat<float,STATE_SIZE,STATE_SIZE>(s_Ak,STATE_SIZE);
            printf("Bk\n");
            printMat<float,STATE_SIZE,CONTROL_SIZE>(s_Bk,STATE_SIZE);
            printf("Qk\n");
            printMat<float,STATE_SIZE,STATE_SIZE>(s_Qk,STATE_SIZE);
            printf("Rk\n");
            printMat<float,CONTROL_SIZE,CONTROL_SIZE>(s_Rk,CONTROL_SIZE);
            printf("qk\n");
            printMat<float,STATE_SIZE, 1>(s_qk,1);
            printf("rk\n");
            printMat<float,CONTROL_SIZE, 1>(s_rk,1);
            printf("Qkp1\n");
            printMat<float,STATE_SIZE,STATE_SIZE>(s_Qkp1,STATE_SIZE);
            printf("qkp1\n");
            printMat<float,STATE_SIZE, 1>(s_qkp1,1);
            printf("integrator error\n");
        }
        __syncthreads();//----------------------------------------------------------------
#endif /* #if DEBUG_MODE */
        
        // Invert Q, Qp1, R 
        loadIdentity<float, STATE_SIZE,STATE_SIZE,CONTROL_SIZE>(
            s_Qk_i, 
            s_Qkp1_i, 
            s_Rk_i
        );
        __syncthreads();//----------------------------------------------------------------
        invertMatrix<float, STATE_SIZE,STATE_SIZE,CONTROL_SIZE,STATE_SIZE>(
            s_Qk, 
            s_Qkp1, 
            s_Rk, 
            s_extra_temp
        );
        __syncthreads();//----------------------------------------------------------------

        // save Qk_i into G (now Ginv) for calculating dz
        gato_memcpy<float>(
            d_G+(blockrow-1)*G_set_size,
            s_Qk_i,
            STATES_SQ
        );

        // save Rk_i into G (now Ginv) for calculating dz
        gato_memcpy<float>( 
            d_G+(blockrow-1)*G_set_size+STATES_SQ,
            s_Rk_i,
            CONTROLS_SQ
        );

        if(blockrow==KNOT_POINTS-1){
            // save Qkp1_i into G (now Ginv) for calculating dz
            gato_memcpy<float>(
                d_G+(blockrow)*G_set_size,
                s_Qkp1_i,
                STATES_SQ
            );
        }
        __syncthreads();//----------------------------------------------------------------

#if DEBUG_MODE
        if(blockrow==1&&GATO_THREAD_ID==0){
            printf("Qk\n");
            printMat<float, STATE_SIZE,STATE_SIZE>(s_Qk_i,STATE_SIZE);
            printf("RkInv\n");
            printMat<float,CONTROL_SIZE,CONTROL_SIZE>(s_Rk_i,CONTROL_SIZE);
            printf("Qkp1Inv\n");
            printMat<float, STATE_SIZE,STATE_SIZE>(s_Qkp1_i,STATE_SIZE);
            printf("\n");
        }
        __syncthreads();//----------------------------------------------------------------
#endif /* #if DEBUG_MODE */


        // Compute -AQ^{-1} in phi
        mat_mat_prod(
            s_phi_k,
            s_Ak,
            s_Qk_i,
            STATE_SIZE, 
            STATE_SIZE, 
            STATE_SIZE, 
            STATE_SIZE
        );
        // for(int i = GATO_THREAD_ID; i < STATES_SQ; i++){
        //     s_phi_k[i] *= -1;
        // }

        __syncthreads();//----------------------------------------------------------------

        // Compute -BR^{-1} in Qkp1
        mat_mat_prod(
            s_Qkp1,
            s_Bk,
            s_Rk_i,
            STATE_SIZE,
            CONTROL_SIZE,
            CONTROL_SIZE,
            CONTROL_SIZE
        );

        __syncthreads();//----------------------------------------------------------------

        // compute Q_{k+1}^{-1}q_{k+1} - IntegratorError in gamma
        mat_vec_prod<float, STATE_SIZE, STATE_SIZE>(
            s_Qkp1_i,
            s_qkp1,
            s_gamma_k
        );
        for(unsigned i = GATO_THREAD_ID; i < STATE_SIZE; i += GATO_THREADS_PER_BLOCK){
            s_gamma_k[i] -= d_c[(blockrow*STATE_SIZE)+i];
        }
        __syncthreads();//----------------------------------------------------------------

        // compute -AQ^{-1}q for gamma         temp storage in extra temp
        mat_vec_prod<float, STATE_SIZE, STATE_SIZE>(
            s_phi_k,
            s_qk,
            s_extra_temp
        );
        

        __syncthreads();//----------------------------------------------------------------
        
        // compute -BR^{-1}r for gamma           temp storage in extra temp + states
        mat_vec_prod<float, STATE_SIZE, CONTROL_SIZE>(
            s_Qkp1,
            s_rk,
            s_extra_temp + STATE_SIZE
        );

        __syncthreads();//----------------------------------------------------------------
        
        // gamma = yeah...
        for(unsigned i = GATO_THREAD_ID; i < STATE_SIZE; i += GATO_THREADS_PER_BLOCK){
            s_gamma_k[i] += s_extra_temp[STATE_SIZE + i] + s_extra_temp[i]; 
        }
        __syncthreads();//----------------------------------------------------------------

        // compute AQ^{-1}AT   -   Qkp1^{-1} for theta
        mat_mat_prod(
            s_theta_k,
            s_phi_k,
            s_Ak,
            STATE_SIZE,
            STATE_SIZE,
            STATE_SIZE,
            STATE_SIZE,
            true
        );

        __syncthreads();//----------------------------------------------------------------

#if DEBUG_MODE
        if(blockrow==1&&GATO_THREAD_ID==0){
            printf("this is the A thing\n");
            printMat<float, STATE_SIZE, STATE_SIZE>(s_theta_k, 234);
        }
#endif /* #if DEBUG_MODE */

        for(unsigned i = GATO_THREAD_ID; i < STATES_SQ; i += GATO_THREADS_PER_BLOCK){
            s_theta_k[i] += s_Qkp1_i[i];
        }
        
        __syncthreads();//----------------------------------------------------------------

        // compute BR^{-1}BT for theta            temp storage in QKp1{-1}
        mat_mat_prod(
            s_Qkp1_i,
            s_Qkp1,
            s_Bk,
            STATE_SIZE,
            CONTROL_SIZE,
            STATE_SIZE,
            CONTROL_SIZE,
            true
        );

        __syncthreads();//----------------------------------------------------------------

        for(unsigned i = GATO_THREAD_ID; i < STATES_SQ; i += GATO_THREADS_PER_BLOCK){
            s_theta_k[i] += s_Qkp1_i[i];
        }
        __syncthreads();//----------------------------------------------------------------

        // save phi_k into left off-diagonal of S, 
        store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
            s_phi_k,                        // src             
            d_S,                            // dst             
            0,                              // col
            blockrow,                        // blockrow    
            -1
        );
        __syncthreads();//----------------------------------------------------------------

        // save -s_theta_k main diagonal S
        store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
            s_theta_k,                                               
            d_S,                                                 
            1,                                               
            blockrow,
            -1                                             
        );          
        __syncthreads();//----------------------------------------------------------------

#if BLOCK_J_PRECON || SS_PRECON
    // invert theta
    loadIdentity<float,STATE_SIZE>(s_thetaInv_k);
    __syncthreads();//----------------------------------------------------------------
    invertMatrix<float,STATE_SIZE>(s_theta_k, s_extra_temp);
    __syncthreads();//----------------------------------------------------------------


    // save thetaInv_k main diagonal PhiInv
    store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
        s_thetaInv_k, 
        d_Pinv,
        1,
        blockrow,
        -1
    );
#else /* BLOCK_J_PRECONDITIONER || SS_PRECONDITIONER  */

    // save 1 / diagonal to PhiInv
    for(int i = GATO_THREAD_ID; i < STATE_SIZE; i+=GATO_THREADS_PER_BLOCK){
        d_Pinv[blockrow*(3*STATES_SQ)+STATES_SQ+i*STATE_SIZE+i]= 1 / d_S[blockrow*(3*STATES_SQ)+STATES_SQ+i*STATE_SIZE+i]; 
    }
#endif /* BLOCK_J_PRECONDITIONER || SS_PRECONDITIONER  */
    

    __syncthreads();//----------------------------------------------------------------

    // save gamma_k in gamma
    for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
        unsigned offset = (blockrow)*STATE_SIZE + ind;
        d_gamma[offset] = s_gamma_k[ind]*-1;
    }

    __syncthreads();//----------------------------------------------------------------

    //transpose phi_k
    loadIdentity<float,STATE_SIZE>(s_Ak);
    __syncthreads();//----------------------------------------------------------------
    mat_mat_prod(s_Qkp1,s_Ak,s_phi_k,STATE_SIZE,STATE_SIZE,STATE_SIZE,STATE_SIZE,true);
    __syncthreads();//----------------------------------------------------------------

    // save phi_k_T into right off-diagonal of S,
    store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
        s_Qkp1,                        // src             
        d_S,                            // dst             
        2,                              // col
        blockrow-1,                      // blockrow    
        -1
    );

    __syncthreads();//----------------------------------------------------------------
    }

}

__global__
void gato_form_schur_jacobi(float *d_G,
                            float *d_C,
                            float *d_g,
                            float *d_c,
                            float *d_S,
                            float *d_Pinv, 
                            float *d_gamma){


    const unsigned s_temp_size =    8 * STATE_SIZE*STATE_SIZE+   
                                    7 * STATE_SIZE+ 
                                    STATE_SIZE * CONTROL_SIZE+
                                     3 * CONTROL_SIZE + 2 * CONTROL_SIZE* CONTROL_SIZE + 3;
                                /// TODO: determine actual shared mem size needed
    
    __shared__ float s_temp[ s_temp_size ];

    for(unsigned blockrow=GATO_BLOCK_ID; blockrow<KNOT_POINTS; blockrow+=GATO_NUM_BLOCKS){

        gato_form_schur_jacobi_inner(
            d_G,
            d_C,
            d_g,
            d_c,
            d_S,
            d_Pinv,
            d_gamma,
            s_temp,
            blockrow
        );
    
    }
}

#if SS_PRECON
__device__
void gato_form_ss_inner(float *d_S, float *d_Pinv, float *d_gamma, float *s_temp, unsigned blockrow){
    
    //  STATE OF DEVICE MEM
    //  S:      -Q0_i in spot 00, phik left off-diagonal, thetak main diagonal
    //  Phi:    -Q0 in spot 00, theta_invk main diagonal
    //  gamma:  -Q0_i*q0 spot 0, gammak


    // GOAL SPACE ALLOCATION IN SHARED MEM
    // s_temp  = | phi_k_T | phi_k | phi_kp1 | thetaInv_k | thetaInv_kp1 | thetaInv_km1 | PhiInv_R | PhiInv_L | scratch
    float *s_phi_k = s_temp;
    float *s_phi_kp1_T = s_phi_k + STATES_SQ;
    float *s_thetaInv_k = s_phi_kp1_T + STATES_SQ;
    float *s_thetaInv_km1 = s_thetaInv_k + STATES_SQ;
    float *s_thetaInv_kp1 = s_thetaInv_km1 + STATES_SQ;
    float *s_PhiInv_k_R = s_thetaInv_kp1 + STATES_SQ;
    float *s_PhiInv_k_L = s_PhiInv_k_R + STATES_SQ;
    float *s_scratch = s_PhiInv_k_L + STATES_SQ;

    // load phi_kp1_T
    if(blockrow!=KNOT_POINTS){
        load_block_bd<float, STATE_SIZE, KNOT_POINTS>(
            d_S,                // src
            s_phi_kp1_T,        // dst
            0,                  // block column (0, 1, or 2)
            blockrow+1,          // block row
            true                // transpose
        );
    }
    
    __syncthreads();//----------------------------------------------------------------

    // load phi_k
    if(blockrow!=0){
        load_block_bd<float, STATE_SIZE, KNOT_POINTS>(
            d_S,
            s_phi_k,
            0,
            blockrow
        );
    }
    
    __syncthreads();//----------------------------------------------------------------


    // load thetaInv_k
    load_block_bd<float, STATE_SIZE, KNOT_POINTS>(
        d_Pinv,
        s_thetaInv_k,
        1,
        blockrow
    );

    __syncthreads();//----------------------------------------------------------------

    // load thetaInv_km1
    if(blockrow!=0){
        load_block_bd<float, STATE_SIZE, KNOT_POINTS>(
            d_Pinv,
            s_thetaInv_km1,
            1,
            blockrow-1
        );
    }

    __syncthreads();//----------------------------------------------------------------

    // load thetaInv_kp1
    if(blockrow!=KNOT_POINTS){
        load_block_bd<float, STATE_SIZE, KNOT_POINTS>(
            d_Pinv,
            s_thetaInv_kp1,
            1,
            blockrow+1
        );
    }
    

    __syncthreads();//----------------------------------------------------------------

    if(blockrow!=0){

        // compute left off diag    
        mat_mat_prod(
            s_scratch,
            s_thetaInv_k,
            s_phi_k,
            STATE_SIZE,
            STATE_SIZE,
            STATE_SIZE,
            STATE_SIZE                           
        );
        __syncthreads();//----------------------------------------------------------------
        mat_mat_prod(
            s_PhiInv_k_L,
            s_scratch,
            s_thetaInv_km1,
            STATE_SIZE,
            STATE_SIZE,
            STATE_SIZE,
            STATE_SIZE
        );
        __syncthreads();//----------------------------------------------------------------

        // store left diagonal in Phi
        store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
            s_PhiInv_k_L, 
            d_Pinv,
            0,
            blockrow,
            -1
        );
        __syncthreads();//----------------------------------------------------------------
    }


    if(blockrow!=KNOT_POINTS){

        /// TODO: transpose here instead of recalc
        // calculate Phi right diag
        mat_mat_prod(
            s_scratch,
            s_thetaInv_k,
            s_phi_kp1_T,
            STATE_SIZE,                           
            STATE_SIZE,                           
            STATE_SIZE,                           
            STATE_SIZE                           
        );
        __syncthreads();//----------------------------------------------------------------
        mat_mat_prod(
            s_PhiInv_k_R,
            s_scratch,
            s_thetaInv_kp1,
            STATE_SIZE,
            STATE_SIZE,
            STATE_SIZE,
            STATE_SIZE
        );
        __syncthreads();//----------------------------------------------------------------

        // store Phi right diag
        store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
            s_PhiInv_k_R, 
            d_Pinv,
            2,
            blockrow,
            -1
        );

    }
}


__global__
void gato_form_ss(float *d_S, float *d_Pinv, float *d_gamma){
    
    const unsigned s_temp_size = 9 * STATES_SQ;
    // 8 * states^2
    // scratch space = states^2

    __shared__ float s_temp[ s_temp_size ];

    for(unsigned ind=GATO_BLOCK_ID; ind<KNOT_POINTS; ind+=GATO_NUM_BLOCKS){
        gato_form_ss_inner(
            d_S,
            d_Pinv,
            d_gamma,
            s_temp,
            ind
        );
    }
}
#endif /* #if SS_PRECON */

/// TODO: make more parallel
__device__
void csr_to_custom_G(int *rowan, int *col, float *val, float *d_G, float rho){

    
    int row_start, row_end, in_set_row, set_offset, in_set_col;
    unsigned row, step, iter;
    
    row = GATO_BLOCK_ID*GATO_THREADS_PER_BLOCK+GATO_THREAD_ID;
    step = GATO_THREADS_PER_BLOCK*GATO_NUM_BLOCKS;
    
    for(; row < (STATES_S_CONTROLS)*KNOT_POINTS-CONTROL_SIZE; row +=step){    

        row_start = rowan[row];
        row_end = rowan[row+1];

        in_set_row = row % (STATE_SIZE+CONTROL_SIZE);
        set_offset = (row / (STATE_SIZE + CONTROL_SIZE)) * (STATES_SQ + CONTROLS_SQ);

        for(iter=row_start; iter<row_end; iter++){

            in_set_col = col[iter] % (STATE_SIZE+CONTROL_SIZE);

            if( in_set_col < STATE_SIZE){
                d_G[set_offset + in_set_col * STATE_SIZE +in_set_row] = val[iter] + (col[iter]==row)*rho;
            }
            else{
                d_G[set_offset + STATES_SQ + (in_set_col - STATE_SIZE) * CONTROL_SIZE + (in_set_row - STATE_SIZE)] = val[iter] + (col[iter]==row)*rho;
            }
        }
    }
}

// checked
__device__
void csr_to_custom_C(int * d_C_row, int * d_C_col, float * d_C_val, float *d_C){

    /*
    out size   (STATES_SQ+STATES_P_CONTROLS)*(KNOT_POINTS-1)*sizeof(float)

    output must be initialized to zeroes
    */

    int col, row_start, row_end, block_row;
    unsigned row, step, iter;
    
    row = GATO_BLOCK_ID*GATO_THREADS_PER_BLOCK+GATO_THREAD_ID;
    step = GATO_THREADS_PER_BLOCK*GATO_NUM_BLOCKS;
    
    // step through rows
    for(; row < STATE_SIZE*KNOT_POINTS; row +=step){

        if(row < STATE_SIZE){continue;}

        row_start = d_C_row[row];
        row_end = d_C_row[row+1];

        block_row = (row / STATE_SIZE)-1;

        for(iter=row_start; iter<row_end; iter++){
            
            col = d_C_col[iter];
            if((col/(STATE_SIZE+CONTROL_SIZE))>block_row){continue;}

            d_C[ block_row*(STATES_SQ+STATES_P_CONTROLS)
                            + (col % (STATE_SIZE+CONTROL_SIZE)) * STATE_SIZE
                            + (row % (STATE_SIZE)) ] = d_C_val[iter];

        }
    }
}

__global__
void gato_convert_kkt_format(int *d_G_row, int *d_G_col, float *d_G_val,
                             int *d_C_row, int *d_C_col, float *d_C_val,
                             float *d_G, float *d_C, float rho){
    
    // convert C to custom dense format
    csr_to_custom_C(d_C_row, d_C_col, d_C_val, d_C);

    // convert G to custom dense format
    csr_to_custom_G(d_G_row, d_G_col, d_G_val, d_G, rho);

}

__device__
void gato_compute_dz_inner(float *d_Ginv_dense, float *d_C_dense, float *d_g_val, float *d_lambda, float *d_dz, float *s_mem, int setrow){

    const unsigned set = setrow/2;
    
    if(setrow%2){ // control row
        // shared mem config
        //    Rkinv |   BkT
        //      C^2  |  S*C

        float *s_Rk_i = s_mem;
        float *s_BkT = s_Rk_i + CONTROLS_SQ;
        float *s_scratch = s_BkT + STATES_P_CONTROLS;

        // load Rkinv from G
        gato_memcpy(s_Rk_i, 
                    d_Ginv_dense+set*(STATES_SQ+CONTROLS_SQ)+STATES_SQ, 
                    CONTROLS_SQ);

        // load Bk from C
        gato_memcpy(s_BkT,
                    d_C_dense+set*(STATES_SQ+STATES_P_CONTROLS)+STATES_SQ,
                    STATES_P_CONTROLS);

        __syncthreads();

        // // compute BkT*lkp1
        gato_ATx(s_scratch,
                 s_BkT,
                 d_lambda+(set+1)*STATE_SIZE,
                 STATE_SIZE,
                 CONTROL_SIZE);
        __syncthreads();

        // subtract from rk
        gato_vec_dif(s_scratch,
                     d_g_val+set*(STATES_S_CONTROLS)+STATE_SIZE,
                     s_scratch,
                     CONTROL_SIZE);
        __syncthreads();

        // multiply Rk_i*scratch in scratch + C
        mat_vec_prod<float, CONTROL_SIZE, CONTROL_SIZE>(s_Rk_i,
                                                        s_scratch,
                                                        s_scratch+CONTROL_SIZE);
        __syncthreads();
        
        // store in d_dz
        gato_memcpy<float>(d_dz+set*(STATES_S_CONTROLS)+STATE_SIZE,
                           s_scratch+CONTROL_SIZE,
                           CONTROL_SIZE);

    }
    else{   // state row

        float *s_Qk_i = s_mem;
        float *s_AkT = s_Qk_i + STATES_SQ;
        float *s_scratch = s_AkT + STATES_SQ;
        
        // shared mem config
        //    Qkinv |  AkT | scratch
        //      S^2     S^2

        /// TODO: error check
        // load Qkinv from G
        gato_memcpy(s_Qk_i, 
                    d_Ginv_dense+set*(STATES_SQ+CONTROLS_SQ), 
                    STATES_SQ);

        // load Ak from C
        gato_memcpy(s_AkT,
                    d_C_dense+set*(STATES_SQ+STATES_P_CONTROLS),
                    STATES_SQ);
        __syncthreads();
                    
        // // compute AkT*lkp1 in scratch
        gato_ATx(s_scratch,
                 s_AkT,
                 d_lambda+(set+1)*STATE_SIZE,
                 STATE_SIZE,
                 STATE_SIZE);
        __syncthreads();

        // add lk to scratch
        gato_vec_sum(s_scratch,     // out
                     d_lambda+set*STATE_SIZE,
                     s_scratch,
                     STATE_SIZE);
        __syncthreads();

        // subtract from qk in scratch
        gato_vec_dif(s_scratch,
                     d_g_val+set*(STATES_S_CONTROLS),
                     s_scratch,
                     STATE_SIZE);
        __syncthreads();
        
        
        // multiply Qk_i(scratch) in Akt
        mat_vec_prod<float, STATE_SIZE, STATE_SIZE>(s_Qk_i,
                                                    s_scratch,
                                                    s_AkT);
        __syncthreads();

        // store in dz
        gato_memcpy<float>(d_dz+set*(STATES_S_CONTROLS),
                           s_AkT,
                           STATE_SIZE);
    }
}

__global__
void gato_compute_dz(float *d_G_dense, float *d_C_dense, float *d_g_val, float *d_lambda, float *d_dz){
    
    // const unsigned s_mem_size = max(2*CONTROL_SIZE, STATE_SIZE);

    __shared__ float s_mem[2*STATES_SQ+STATE_SIZE]; 

    for(int ind = GATO_BLOCK_ID; ind < 2*KNOT_POINTS-1; ind+=GATO_NUM_BLOCKS){
        gato_compute_dz_inner(d_G_dense, d_C_dense, d_g_val, d_lambda, d_dz, s_mem, ind);
    }
}

/*******************************************************************************
 *                           Interface Functions                                *
 *******************************************************************************/

void form_schur(int *d_G_row, int *d_G_col, float *d_G_val, float *d_G_dense,
		int *d_C_row, int *d_C_col, float *d_C_val, float *d_C_dense,
		float *d_g_val,
        float *d_c_val,
        float *d_S, float *d_Pinv, float *d_gamma, float rho=0){
    
    

    dim3 launch_block(((max(STATES_SQ, CONTROLS_SQ) / 32) + 1)*32);
    dim3 launch_grid(KNOT_POINTS);
    

    // convert G, C, g into custom formats
    gato_convert_kkt_format<<<launch_grid, launch_block>>>(d_G_row,d_G_col,d_G_val,
                                                               d_C_row,d_C_col,d_C_val,
                                                               d_G_dense, d_C_dense, rho);
    gpuErrchk( cudaPeekAtLastError() );

#if GATO_PRINTING

    float G_copy[KKT_G_DENSE_SIZE_BYTES];
    float C_copy[KKT_C_DENSE_SIZE_BYTES];
    
    gpuErrchk( cudaMemcpy(G_copy, d_G_dense, KKT_G_DENSE_SIZE_BYTES, cudaMemcpyDeviceToHost));
    gpuErrchk( cudaMemcpy(C_copy, d_C_dense, KKT_C_DENSE_SIZE_BYTES, cudaMemcpyDeviceToHost));

    gpuErrchk( cudaDeviceSynchronize() );

    std::cout << "G\n";
    for(int i = 0; i < KKT_G_DENSE_SIZE_BYTES/sizeof(float); i++){
        if(i%(STATES_SQ+CONTROLS_SQ) == 0)
            std::cout << "\n";
        std::cout << G_copy[i] << " ";
    }
    std::cout << "\n";
    std::cout << "\nC\n";
    for(int i = 0; i < KKT_C_DENSE_SIZE_BYTES/sizeof(float); i++){
        if(i%(STATES_SQ+STATES_P_CONTROLS) == 0)
            std::cout << "\n";
        std::cout << C_copy[i] << " ";
    }
    std::cout << "thhasdfasdf";
#endif /* GATO_PRINTING */
    
    // form Schur, Jacobi
    gato_form_schur_jacobi<<<launch_grid, launch_block>>>(d_G_dense, d_C_dense, d_g_val, d_c_val, d_S, d_Pinv, d_gamma);
    gpuErrchk( cudaPeekAtLastError() );

#if GATO_PRINTING
    
    float G_dense_copy[KKT_G_DENSE_SIZE_BYTES];
    memset((void *)G_dense_copy, 0, KKT_G_DENSE_SIZE_BYTES);
    gpuErrchk( cudaMemcpy(G_dense_copy, d_G_dense, KKT_G_DENSE_SIZE_BYTES, cudaMemcpyDeviceToHost));
    
    gpuErrchk( cudaDeviceSynchronize() );

    std::cout << "Ginv\n";
    for(int i = 0; i < KKT_G_DENSE_SIZE_BYTES/sizeof(float); i++){
        if(i%(STATES_SQ+CONTROLS_SQ) == 0)
            std::cout << "\n";
        std::cout << G_dense_copy[i] << " ";
    }
    std::cout << std::endl;
#endif /* GATO_PRINTING */


#if SS_PRECON
    
    gato_form_ss<<<launch_grid, launch_block>>>(d_S, d_Pinv, d_gamma);
    gpuErrchk( cudaPeekAtLastError() );

#endif  /* #if SS_PRECONDITIONER */

#if GATO_PRINTING

    float S_copy[3*STATES_SQ*KNOT_POINTS];
    float Pinv_copy[3*STATES_SQ*KNOT_POINTS];
    float gamma_copy[STATE_SIZE*KNOT_POINTS];
    gpuErrchk( cudaMemcpy(S_copy, d_S, 3*STATES_SQ*KNOT_POINTS*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk( cudaMemcpy(Pinv_copy, d_Pinv, 3*STATES_SQ*KNOT_POINTS*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk( cudaMemcpy(gamma_copy, d_gamma, STATE_SIZE*KNOT_POINTS*sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk( cudaDeviceSynchronize() );

    printf("S\n");
    print_bd_matrix<float>(S_copy, STATE_SIZE, KNOT_POINTS);
    printf("\n\nPinv\n");
    print_bd_matrix<float>(Pinv_copy, STATE_SIZE, KNOT_POINTS);
    printf("\n\ngamma\n");
    for(int i = 0; i < KNOT_POINTS; i++){
        for(int j = 0; j < STATE_SIZE; j++){
            printf("%f\t", gamma_copy[i*STATE_SIZE+j]);
        }
        printf("\n");
    }
    printf("\n");
#endif  /* #if GATO_PRINTING */
}


void compute_dz(float *d_G_dense, float *d_C_dense, float *d_g_val, float *d_lambda, float *d_dz){
    
    dim3 launch_block(((max(STATES_SQ, CONTROLS_SQ) / 32) + 1)*32);
    dim3 launch_grid(KNOT_POINTS);

    gato_compute_dz<<<launch_grid, launch_block>>>(d_G_dense, d_C_dense, d_g_val, d_lambda, d_dz);
    gpuErrchk( cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );

#if DEBUG_MODE
    gpuErrchk( cudaDeviceSynchronize());
#endif /* DEBUG_MODE */
}
