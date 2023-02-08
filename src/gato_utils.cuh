#ifndef GATO_UTILS_CUH
#define GATO_UTILS_CUH

#include <iostream>
#include <iomanip>
#include "../include/gato_defines.h"
#include "../include/types.h"

#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;


/*******************************************************************************
*                              loady things                                    *
*******************************************************************************/

/// TODO: this is really stupid but has to be done for now
/// TODO: error check?
template <typename T>
__device__
void gato_memcpy(T *dst, T *src, unsigned size_Ts){
    unsigned ind;
    for(ind=GATO_THREAD_ID; ind < size_Ts; ind+=GATO_THREADS_PER_BLOCK){
        dst[ind] = src[ind];
    }
}

// just negates the val lol
template <typename T>
__device__
void gato_nmemcpy(T *dst, T *src, unsigned size_Ts){
    unsigned ind;
    for(ind=GATO_THREAD_ID; ind < size_Ts; ind+=GATO_THREADS_PER_BLOCK){
        dst[ind] = -src[ind];
    }
}


// src is a B_DIM X B_DIM column-major matrix
// dst is a diagonal-format, block-tri, column_major, M_DIM*B_DIM X M_DIM*B_DIM matrix where col = 0, 1, 2 indicates left-diag, main-diag, right-diag
// offset might be needed but I don't think so
// multiplier multiplies src before storing

template <typename T, unsigned B_DIM, unsigned M_DIM>
__device__
void store_block_bd(T *src, T *dst, unsigned col, unsigned BLOCKNO, int multiplier=1){
    
    unsigned block_row_offset, block_col_offset, ind;

    assert(col<3);


    block_row_offset = BLOCKNO * (3 * B_DIM * B_DIM);
    block_col_offset = col*B_DIM*B_DIM;


    if(multiplier==1){

        gato_memcpy<T>(
            dst+block_row_offset+block_col_offset,
            src,
            B_DIM*B_DIM
        );

    }
    else{
        
        for(ind=GATO_THREAD_ID; ind<B_DIM*B_DIM; ind+=GATO_THREADS_PER_BLOCK){
            dst[block_row_offset + block_col_offset + ind] = src[ind] * multiplier;
        }

    }
}

template <typename T, unsigned m, unsigned n>
__device__
void printMat(float *mat, int asdf){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            printf("%f ", mat[j*m+i]);
        }
        printf("\n");
    }
}

template <typename T, unsigned B_DIM, unsigned M_DIM>
__device__
void load_block_bd(T *src, T *dst, unsigned bcol, unsigned brow, bool transpose=false){
    
    // EMRE assert this
    if(bcol > 2 || brow > M_DIM-1)
        return;
    

    unsigned block_row_offset, block_col_offset;

    block_row_offset = brow * (3 * B_DIM * B_DIM);
    block_col_offset = bcol*B_DIM*B_DIM;

    if(!transpose){

        gato_memcpy<T>(
            dst,
            src+block_row_offset+block_col_offset,
            B_DIM*B_DIM
        );

    }
    else{

        unsigned ind, transpose_col, transpose_row;

        for(ind=GATO_THREAD_ID; ind<B_DIM*B_DIM; ind+=GATO_THREADS_PER_BLOCK){
            transpose_col = ind%B_DIM * B_DIM;
            transpose_row = ind/B_DIM;
            dst[transpose_col + transpose_row] = src[block_row_offset + block_col_offset + ind];    
        }
    }
}

template <typename T, unsigned BLOCK_DIM>
__device__
void loadBlockTriDiagonal_offDiagonal(T *s_var, T *d_var_b, cgrps::thread_block block, cgrps::grid_group grid){
    // Need to load b also now
    for (unsigned ind = GATO_THREAD_ID; ind < BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
        s_var[ind + BLOCK_DIM] = *(d_var_b + ind); 
    }
    // if first block just want b and b+1 (and already have b)
    if(GATO_LEAD_BLOCK){
        for (unsigned ind = GATO_THREAD_ID; ind < BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
            s_var[ind + 2*BLOCK_DIM] = *(d_var_b + BLOCK_DIM + ind); // just b+1
        }

    }
    // if last block just want b-1 and b (and already have b)
    else if (GATO_LAST_BLOCK){
        for (unsigned ind = GATO_THREAD_ID; ind < BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
            s_var[ind] = *(d_var_b - BLOCK_DIM + ind); // just b-1
        }

    }
    // else want b-1 and b and b+1 (and already have b)
    else{
        for (unsigned ind = GATO_THREAD_ID; ind < 2*BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
            T *dst, *src;
            if (ind < BLOCK_DIM){dst = s_var + ind;       	  src = d_var_b - BLOCK_DIM + ind;} // b-1
            else		  		{dst = s_var + BLOCK_DIM + ind; src = d_var_b + ind;} // b+1
            *dst = *src;
        }
    }
}

template <typename T, unsigned BLOCK_DIM>
__device__ 
void matVecMultBlockTriDiagonal(T *s_dst, T *s_mat, T *s_vec, cgrps::thread_block block, cgrps::grid_group grid){
    // First or Last block only 2 mults (var and either var+1 or var-1)
    if(GATO_LEAD_BLOCK){
        for (unsigned r = GATO_THREAD_ID; r < BLOCK_DIM; r += GATO_THREADS_PER_BLOCK){
            T val = static_cast<T>(0);
            for(unsigned c = 0; c < 2*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM*BLOCK_DIM + BLOCK_DIM * c + r] * s_vec[c + BLOCK_DIM]; // var and var+1
            }
            s_dst[r] = val;
        }
    }
    else if (GATO_LAST_BLOCK){
        for (unsigned r = GATO_THREAD_ID; r < BLOCK_DIM; r += GATO_THREADS_PER_BLOCK){
            T val = static_cast<T>(0);
            for(unsigned c = 0; c < 2*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM * c + r] * s_vec[c]; // var and var-1
            }
            s_dst[r] = val;
        }
    }
    // else 3 mults
    else{
        for (unsigned r = GATO_THREAD_ID; r < BLOCK_DIM; r += GATO_THREADS_PER_BLOCK){
            T val = static_cast<T>(0);
            for(unsigned c = 0; c < 3*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM * c + r] * s_vec[c];
            }
            s_dst[r] = val;
        }
    }
}

template <typename T, unsigned BLOCK_DIM, unsigned N>
__device__
void loadBlockTriDiagonal_offDiagonal_fixed(T *s_var, T *d_var_b, unsigned block_row, cgrps::thread_block block, cgrps::grid_group grid){
    // Need to load b also now
    for (unsigned ind = GATO_THREAD_ID; ind < BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
        s_var[ind + BLOCK_DIM] = *(d_var_b + ind); 
    }
    // if first block just want b and b+1 (and already have b)
    if(block_row == 0){
        for (unsigned ind = GATO_THREAD_ID; ind < BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
            s_var[ind + 2*BLOCK_DIM] = *(d_var_b + BLOCK_DIM + ind); // just b+1
        }

    }
    // if last block just want b-1 and b (and already have b)
    else if (block_row == N){
        for (unsigned ind = GATO_THREAD_ID; ind < BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
            s_var[ind] = *(d_var_b - BLOCK_DIM + ind); // just b-1
        }

    }
    // else want b-1 and b and b+1 (and already have b)
    else{
        for (unsigned ind = GATO_THREAD_ID; ind < 2*BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
            T *dst, *src;
            if (ind < BLOCK_DIM){dst = s_var + ind;       	  src = d_var_b - BLOCK_DIM + ind;} // b-1
            else		  		{dst = s_var + BLOCK_DIM + ind; src = d_var_b + ind;} // b+1
            *dst = *src;
        }
    }
}

template <typename T, unsigned BLOCK_DIM, unsigned N>
__device__ 
void matVecMultBlockTriDiagonal_fixed(T *s_dst, T *s_mat, T *s_vec, unsigned block_row, cgrps::thread_block block, cgrps::grid_group grid){
    // First or Last block only 2 mults (var and either var+1 or var-1)
    if(block_row == 0 ){
        for (unsigned r = GATO_THREAD_ID; r < BLOCK_DIM; r += GATO_THREADS_PER_BLOCK){
            T val = static_cast<T>(0);
            for(unsigned c = 0; c < 2*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM*BLOCK_DIM + BLOCK_DIM * c + r] * s_vec[c + BLOCK_DIM]; // var and var+1
            }
            s_dst[r] = val;
        }
    }
    else if (block_row == N){
        for (unsigned r = GATO_THREAD_ID; r < BLOCK_DIM; r += GATO_THREADS_PER_BLOCK){
            T val = static_cast<T>(0);
            for(unsigned c = 0; c < 2*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM * c + r] * s_vec[c]; // var and var-1
            }
            s_dst[r] = val;
        }
    }
    // else 3 mults
    else{
        for (unsigned r = GATO_THREAD_ID; r < BLOCK_DIM; r += GATO_THREADS_PER_BLOCK){
            T val = static_cast<T>(0);
            for(unsigned c = 0; c < 3*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM * c + r] * s_vec[c];
            }
            s_dst[r] = val;
        }
    }
}

template <typename T, unsigned VEC_SIZE>
__device__
void reducePlus(T *dstTemp, cgrps::thread_block block){
    unsigned size_left = VEC_SIZE;
    // loop until only a few values left
    while (size_left > 3){
        // determine if odd_adjust needed and update size
        bool odd_flag = size_left % 2;
        size_left = (size_left - odd_flag)/2; 
        // reduce in half
        for (unsigned ind = GATO_THREAD_ID; ind < size_left; ind += GATO_THREADS_PER_BLOCK){
            dstTemp[ind] += dstTemp[ind + size_left];
        }	
        // add the odd size adjust if needed
        if (GATO_LEAD_THREAD && odd_flag){dstTemp[0] += dstTemp[2*size_left];}
        // sync and repeat
        block.sync();
    }
    // when we get really small sum up what is left
    if (GATO_LEAD_THREAD){
        for(unsigned ind = 1; ind < size_left; ind++){dstTemp[0] += dstTemp[ind];}
    }
}

template <typename T, unsigned VEC_SIZE>
__device__
void dotProd(T *dstTemp, T *vec1, T *vec2, cgrps::thread_block block){
    // first compute temp sums across all threads
    for (unsigned ind = GATO_THREAD_ID; ind < VEC_SIZE; ind += GATO_THREADS_PER_BLOCK){
        dstTemp[ind] = vec1[ind] * vec2[ind];
    }
    block.sync();
    // then reduce
    reducePlus<T,VEC_SIZE>(dstTemp,block);
}

/*******************************************************************************
*                              printy things                                   *
*******************************************************************************/



template <typename T, unsigned M, unsigned N>
__host__ __device__
void print_block(T *A){
    for(unsigned i=0; i<M; i++){
        for(unsigned j=0; j<N; j++){printf("%.4f  ",A[i + M*j]);}
        printf("\n\n");
    }
} 

template <typename T, unsigned B_DIM, unsigned M_DIM>
__host__ __device__
void print_raw_bd_matrix(T *A){

    unsigned row_size, block_size;
    unsigned block_row, block_col, row, col;
    unsigned i, j;

    row_size = 3 * B_DIM * B_DIM;
    block_size = B_DIM * B_DIM;

    for(i=0; i < B_DIM * M_DIM; i++){
        for(j=0; j < B_DIM * 3; j++){

            block_row = i / B_DIM; 
            block_col = j / B_DIM; 
            row = i % B_DIM;
            col = j % B_DIM;
            
            printf("%.4f  ",A[ block_row*row_size + block_col*block_size + col*B_DIM + row ]);
        }
        printf("\n\n");
    }
} 


// __host__ __device__
// template <typename T, unsigned STATE_SIZE>
// void print_raw_shared_matrix(T *A){

//     unsigned row_size, col_size;
//     unsigned i, j;

//     row_size = 3 * STATE_SIZE;
//     col_size = STATE_SIZE;

//     for(int i=0;i<row_size;i++){
//         for(int j=0; j<col_size;j++){
//             printf("%0.4f  ", A[j*STATE_SIZE*STATE_SIZE + i]);
//         }
//         printf("\n");
//     }
// } 


// __host__ __device__
// template <typename T, unsigned ROW_SIZE>
// void print_raw_shared_vector(T *A){
//     unsigned i;

//     for(int i=0;i<ROW_SIZE;i++){
//         printf("%0.4f  ", A[i]); 
//     }
//     printf("\n");
// }


template <typename T>
__host__
void print_bd_matrix(T *A, int block_dim_Ts, int matrix_dim_blocks){

    const unsigned bd = block_dim_Ts;
    const unsigned md = matrix_dim_blocks;
    const unsigned blocksize = bd * bd;
    const unsigned rowsize = 3 * blocksize;
    
    unsigned row, col;
    int blockrow, blockcol;

    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    
    for(row = 0; row < bd * md; row++){
        
        blockrow = row / bd;

        if(blockrow==0){
            for(col=0; col < bd * md; col++){
                
                blockcol = col / bd;
                if(blockcol < 2){ std::cout << A[ (col+bd)*bd + row ] << "\t"; }
                else{ std::cout << static_cast<T>(0) << "\t"; }
            }
        }
        else{
            for(col=0; col < bd * md; col++){
                
                blockcol = col / bd;
                if(blockcol < blockrow-1 || blockcol > blockrow+1){ 
                    std::cout << static_cast<T>(0) << "\t"; 
                }
                else{ 
                    std::cout << A[ rowsize*blockrow + (blockcol-blockrow+1)*blocksize + (col%bd)*bd + row%bd ] << "\t"; 
                }
                
            }
        }
        
        std::cout << "\n\n";
    }
}




/*******************************************************************************
*                            matrix inversion                                  *
*******************************************************************************/
    

// load identity in so memory is [A | I]
template <typename T, unsigned DIM>
__device__ __forceinline__
void loadIdentity(T *A){
    for (unsigned ind = GATO_THREAD_ID; ind < DIM*DIM; ind += GATO_THREADS_PER_BLOCK){
        unsigned r, c;
        r = ind % DIM; 
        c = ind / DIM;
        A[ind] = static_cast<T>(r == c);
    }
}

// load identity in so memory is [V | I]
template <typename T, unsigned DIMA, unsigned DIMB>
__device__ __forceinline__
void loadIdentity(T *A, T *B){
    for (unsigned ind = GATO_THREAD_ID; ind < DIMA*DIMA+DIMB*DIMB; ind += GATO_THREADS_PER_BLOCK){
        unsigned r, c, indAdj; T *V;
        if (ind < DIMA*DIMA){
            indAdj = ind;
            r = indAdj % DIMA; c = indAdj/DIMA; V = A;
        }
        else {
            indAdj = ind - DIMA*DIMA;
            r = indAdj % DIMB; c = indAdj/DIMB; V = B;
        }
        V[indAdj] = static_cast<T>(r == c);
    }
}


// load identity in so memory is [V | I]
template <typename T, unsigned DIMA, unsigned DIMB, unsigned DIMC>
__device__ __forceinline__
void loadIdentity(T *A, T *B, T *C){
    for (unsigned ind = GATO_THREAD_ID; ind < DIMA*DIMA+DIMB*DIMB+DIMC*DIMC; ind += GATO_THREADS_PER_BLOCK){
        unsigned r, c, indAdj; T *V;
        if (ind < DIMA*DIMA){
            indAdj = ind;
            r = indAdj % DIMA; c = indAdj/DIMA; V = A;
        }
        else if (ind < DIMA*DIMA+DIMB*DIMB){
            indAdj = ind - DIMA*DIMA;
            r = indAdj % DIMB; c = indAdj/DIMB; V = B;
        }
        else{
            indAdj = ind - DIMA*DIMA - DIMB*DIMB;
            r = indAdj % DIMC; c = indAdj/DIMC; V = C;
        }
        V[indAdj] = static_cast<T>(r == c);
    }
}


template <typename T, unsigned DIM>
__device__
void invertMatrix(T *A, T *s_temp){ 
// we are going to guassian elimination walking down the matrix (assuming no leading 0s)
// we therefore use the columns in order as the pivot column for each pivot we need to rescale 
// that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple 
// of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
// pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
    for (unsigned pivRC = 0; pivRC < DIM; pivRC++){
        unsigned pivColOffset = pivRC*DIM;
        // save the pivot and pivot column and row
        T pvInv = static_cast<T>(1)/A[pivRC + pivColOffset];
        for (unsigned ind = GATO_THREAD_ID; ind < 2*DIM+1; ind++){
            unsigned AInd;
            if (ind < DIM){AInd = ind + pivColOffset;}
            else{AInd = pivRC + pivColOffset + (ind-DIM)*DIM;}
            s_temp[ind] = A[AInd];
        }
        __syncthreads(); //----------------------
        // make the pivot update
        for (unsigned ind = GATO_THREAD_ID; ind < DIM*(DIM+1); ind += GATO_THREADS_PER_BLOCK){
            unsigned row = ind % DIM; unsigned col = ind / DIM; unsigned colOffset = ind - row;
            // s_temp = orpcvs|prvOld
            if (row == pivRC){A[row + pivColOffset + colOffset] *= pvInv;}
            else{A[row + pivColOffset + colOffset] -= s_temp[row]*pvInv*s_temp[DIM+col];}
        }
    __syncthreads(); //----------------------
    }
}


template <typename T, unsigned DIMA, unsigned DIMB, unsigned MAX_DIM>
__device__
void invertMatrix(T *A, T *B, T *s_temp){

    // now we are going to guassian elimination walking down the matrix (assuming no leading 0s)
    // we therefore use the columns in order as the pivot column for each pivot we need to rescale 
    // that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple 
    // of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
    // pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
    T *s_memA = s_temp; T *s_memB = &s_memA[2*DIMA+1];
    for (unsigned pivRC = 0; pivRC < MAX_DIM; pivRC++){
        bool AActive = pivRC < DIMA; bool BActive = pivRC < DIMB;
        unsigned pivColOffsetA = pivRC*DIMA; unsigned pivColOffsetB = pivRC*DIMB;
        // save the pivot column and row
        for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM; ind++){
            if (AActive && ind < DIMA){s_memA[ind] = A[ind + pivColOffsetA];}
            if (BActive && ind < DIMB){s_memB[ind] = B[ind + pivColOffsetB];}
        }
        for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM+1; ind++){
            if (AActive && ind < DIMA+1){s_memA[ind + DIMA] = A[ind*DIMA + pivRC + pivColOffsetA];}
            if (BActive && ind < DIMB+1){s_memB[ind + DIMB] = B[ind*DIMB + pivRC + pivColOffsetB];}
        }
        __syncthreads(); //----------------------
        // make the pivot update with s_mem = [colA,rowA,colB,rowB,colC,rowC]
        for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM*(MAX_DIM+1); ind += GATO_THREADS_PER_BLOCK){
            if (AActive && ind < DIMA*(DIMA+1)){
                unsigned row = ind % DIMA; unsigned col = ind / DIMA;
                if (row == pivRC){A[pivColOffsetA + ind] /= s_memA[pivRC];}
                else{A[pivColOffsetA + ind] -= s_memA[row]/s_memA[pivRC]*s_memA[DIMA+col];}
            }
            if (BActive && ind < DIMB*(DIMB+1)){
                unsigned row = ind % DIMB; unsigned col = ind / DIMB; 
                if (row == pivRC){B[pivColOffsetB + ind] /= s_memB[pivRC];}
                else{B[pivColOffsetB + ind] -= s_memB[row]/s_memB[pivRC]*s_memB[DIMB+col];}
            }
        }
        __syncthreads(); //----------------------
    }
}

// invert A,B,C assume memory for all is [V | VInv] where both are DIMxDIM and continguous
// relies on s_temp of size [2*DIMA + 2*DIMB + 2*DIMC + 3]
template <typename T, unsigned DIMA, unsigned DIMB, unsigned DIMC, unsigned MAX_DIM>
__device__
void invertMatrix(T *A, T *B, T *C, T *s_temp){

    // now we are going to guassian elimination walking down the matrix (assuming no leading 0s)
    // we therefore use the columns in order as the pivot column for each pivot we need to rescale 
    // that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple 
    // of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
    // pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
    T *s_memA = s_temp; T *s_memB = &s_memA[2*DIMA+1]; T *s_memC = &s_memB[2*DIMB+1];
    for (unsigned pivRC = 0; pivRC < MAX_DIM; pivRC++){
        bool AActive = pivRC < DIMA; bool BActive = pivRC < DIMB; bool CActive = pivRC < DIMC;
        unsigned pivColOffsetA = pivRC*DIMA; unsigned pivColOffsetB = pivRC*DIMB; unsigned pivColOffsetC = pivRC*DIMC;
        // save the pivot column and row
        for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM; ind++){
            if (AActive && ind < DIMA){s_memA[ind] = A[ind + pivColOffsetA];}
            if (BActive && ind < DIMB){s_memB[ind] = B[ind + pivColOffsetB];}
            if (CActive && ind < DIMC){s_memC[ind] = C[ind + pivColOffsetC];}
        }
        for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM+1; ind++){
            if (AActive && ind < DIMA+1){s_memA[ind + DIMA] = A[ind*DIMA + pivRC + pivColOffsetA];}
            if (BActive && ind < DIMB+1){s_memB[ind + DIMB] = B[ind*DIMB + pivRC + pivColOffsetB];}
            if (CActive && ind < DIMC+1){s_memC[ind + DIMC] = C[ind*DIMC + pivRC + pivColOffsetC];}
        }
        __syncthreads(); //----------------------
        // make the pivot update with s_mem = [colA,rowA,colB,rowB,colC,rowC]
        for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM*(MAX_DIM+1); ind += GATO_THREADS_PER_BLOCK){
            if (AActive && ind < DIMA*(DIMA+1)){
                unsigned row = ind % DIMA; unsigned col = ind / DIMA;
                if (row == pivRC){A[pivColOffsetA + ind] /= s_memA[pivRC];}
                else{A[pivColOffsetA + ind] -= s_memA[row]/s_memA[pivRC]*s_memA[DIMA+col];}
            }
            if (BActive && ind < DIMB*(DIMB+1)){
                unsigned row = ind % DIMB; unsigned col = ind / DIMB; 
                if (row == pivRC){B[pivColOffsetB + ind] /= s_memB[pivRC];}
                else{B[pivColOffsetB + ind] -= s_memB[row]/s_memB[pivRC]*s_memB[DIMB+col];}
            }
            if (CActive && ind < DIMC*(DIMC+1)){
                unsigned row = ind % DIMC; unsigned col = ind / DIMC;
                if (row == pivRC){C[pivColOffsetC + ind] /= s_memC[pivRC];}
                else{C[pivColOffsetC + ind] -= s_memC[row]/s_memC[pivRC]*s_memC[DIMC+col];}
            }
        }
        __syncthreads(); //----------------------
    }
}



/*******************************************************************************
*                             matrix operations                                *
*******************************************************************************/


template <typename T, unsigned MAT_ROWS, unsigned MAT_COLS>
__device__
void mat_vec_prod(T *mat, T *vec, T *out){
    
    for(unsigned row=GATO_THREAD_ID; row<MAT_ROWS; row+=GATO_THREADS_PER_BLOCK){
        T res = static_cast<T>(0);
        for (unsigned col = 0; col < MAT_COLS; col++){
            res += mat[row + col*MAT_ROWS] * vec[col];
        }
        out[row] = res;
    }
}

///TODO: this could be more better
__device__
void mat_mat_prod(float *out, float *mat_A, float *mat_B, int A_rows, int A_cols, int B_rows, int B_cols, bool transposeB = false){

    if(!transposeB){

        assert(A_cols==B_rows);

        unsigned ind, thing;
        unsigned maxind = A_rows*B_cols;
        float res;
        int row, col;

        for(ind=GATO_THREAD_ID; ind<maxind; ind+=GATO_THREADS_PER_BLOCK){
            // ind x takes row x/A_cols and col x%b_rows
            res = 0;
            row = ind % A_rows;
            col = ind / A_rows;

            for(thing=0; thing<A_cols; thing++){
                res += mat_A[thing*A_rows+row] * mat_B[col*B_rows+thing];
            }

            out[col*A_rows+row] = res;

        } 
    }
    else{                       // transpose matrix B

        assert(A_cols==B_cols);

        unsigned ind, thing;
        unsigned maxind = A_rows*B_rows;
        float res;
        int row, col;

        for(ind=GATO_THREAD_ID; ind<maxind; ind+=GATO_THREADS_PER_BLOCK){
            // ind x takes row x/A_cols and col x%b_rows
            res = 0;
            row = ind % A_rows;
            col = ind / A_rows;

            for(thing=0; thing<A_cols; thing++){
                res += mat_A[thing*A_rows+row] * mat_B[thing*B_rows+col];
            }

            out[col*A_rows+row] = res;

        } 

    }
}


// // multiplies the transpose of mat with vec, m and n correspond to rows and cols in NON-transposed mat
// /// TODO: totally unchecked 
__device__
void gato_ATx(float *out, float *mat, float *vec, int m, int n){

    float res;
    int ind, thing;

    for(ind=GATO_THREAD_ID; ind < n; ind +=GATO_THREADS_PER_BLOCK){

        res = 0;
        for(thing=0; thing<m; thing++){
            res += mat[ind*m+thing] * vec[thing];
        }

        out[ind] = res;
    }
}

__device__
void gato_vec_dif(float *out, float *vec1, float *vec2, int size){
    for(int i = GATO_THREAD_ID; i < size; i+= GATO_THREADS_PER_BLOCK){
        out[i] = vec1[i] - vec2[i];
    }
}

__device__
void gato_vec_sum(float *out, float *vec1, float *vec2, int size){
    for(int i = GATO_THREAD_ID; i < size; i+= GATO_THREADS_PER_BLOCK){
        out[i] = vec1[i] + vec2[i];
    }
}
/*******************************************************************************
*                              format conversion                               *
*******************************************************************************/

/*   convert csr format to custom block-diagonal-fmt */
__device__
void csr_to_bd(csr_t*csrmat,
                float  *bdmat,
                unsigned bdim,
                unsigned mdim){
    
    int col, row_start, row_end, bd_block_row, bd_block_col, bd_row, bd_col;
    float  val;
    unsigned row, iter;

    
    for(row = GATO_BLOCK_ID*GATO_THREADS_PER_BLOCK+GATO_THREAD_ID; row < csrmat->m; row +=GATO_THREADS_PER_BLOCK*GATO_NUM_BLOCKS){    

        row_start = csrmat->row_ptr[row];
        row_end = csrmat->row_ptr[row+1];

        for(iter=row_start; iter<row_end; iter++){
            col = csrmat->col_ind[iter];
            val = csrmat->val[iter];

            bd_block_row = ( row / bdim );                     // block row
            bd_block_col = ( col / bdim ) + 1 - bd_block_row;  // block col
            bd_col = col % bdim;
            bd_row = row % bdim;

            bdmat[ bd_block_row*3*bdim*bdim + bd_block_col*bdim*bdim + bd_col*bdim + bd_row] = val;
        }
    }

}

__device__
void csr_to_std(csr_t *csrmat,
                float  *stdmat){
    
    int col, row_start, row_end;
    float  val;
    unsigned row, step, iter;
    
    row = GATO_BLOCK_ID*GATO_THREADS_PER_BLOCK+GATO_THREAD_ID;
    step = GATO_THREADS_PER_BLOCK*GATO_NUM_BLOCKS;
    
    for(; row < csrmat->m; row +=step){    

        row_start = csrmat->row_ptr[row];
        row_end = csrmat->row_ptr[row+1];

        for(iter=row_start; iter<row_end; iter++){
            col = csrmat->col_ind[iter];
            val = csrmat->val[iter];

            stdmat[col*csrmat->m + row] = val;
        }
    }
}


__device__
void bd_to_csr(float  *bdmat,
                csr_t *csrmat,
                unsigned bdim,
                unsigned mdim){

    int row, col, csr_row_offset, basic_col_offset, bd_block_row, bd_block_col, bd_col, bd_row, bd_row_len;
    unsigned iter, bd_offset;

    // each thread takes one row
    for(row = GATO_BLOCK_ID*GATO_THREADS_PER_BLOCK+GATO_THREAD_ID; row < csrmat->m; row += GATO_THREADS_PER_BLOCK*GATO_NUM_BLOCKS){

        bd_block_row = row/bdim;

        // row_len
        if(bd_block_row==0 || bd_block_row==mdim-1){
            bd_row_len = 2*bdim;
        }
        else{
            bd_row_len = 3*bdim;
        }

        // set row_ptr
        if(bd_block_row==0){                        // first block
            csr_row_offset = row*bd_row_len;
            basic_col_offset = 0;

            csrmat->row_ptr[row+1] = csr_row_offset+bd_row_len;
            if(row==0){
                csrmat->row_ptr[row] = 0;
            }
        }
        else if(bd_block_row==mdim-1){              // last block
            csr_row_offset = 2*bdim*bdim+(mdim-2)*3*bdim*bdim+(row%bdim)*bd_row_len;
            basic_col_offset = (bd_block_row-1)*bdim;

            csrmat->row_ptr[row+1] = csr_row_offset+bd_row_len;
        }
        else{
            csr_row_offset = 2*bdim*bdim+(row-bdim)*bd_row_len;
            basic_col_offset = (bd_block_row-1)*bdim;

            csrmat->row_ptr[row+1] = csr_row_offset+bd_row_len;
        }

        for(iter=0; iter<bd_row_len; iter++){

            col = basic_col_offset+iter;
            bd_block_row = ( row / bdim );                     // block row
            bd_block_col = ( col / bdim ) + 1 - bd_block_row;  // block col
            bd_col = col % bdim;
            bd_row = row % bdim;

            bd_offset = bd_block_row*3*bdim*bdim + bd_block_col*bdim*bdim + bd_col*bdim + bd_row;
            
            csrmat->col_ind[csr_row_offset+iter] = col;
            csrmat->val[csr_row_offset+iter] = bdmat[bd_offset];
        }

        if(row==csrmat->m-1){
            csrmat->nnz = STATES_SQ*3*KNOT_POINTS;
        }

    }
}



/*******************************************************************************
*                                    other                                     *
*******************************************************************************/


template <typename T>
int check_sms(void* kernel, dim3 block){
    int dev = 0;
    cudaDeviceProp deviceProp; 
    cudaGetDeviceProperties(&deviceProp, dev);
    int supportsCoopLaunch = 0; 
    gpuErrchk(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev));
    if(!supportsCoopLaunch){
        printf("[Error] Device does not support Cooperative Threads -- this code will fail!\n");
        exit(5);
    }
    int numProcs = static_cast<T>(deviceProp.multiProcessorCount); 
    int numBlocksPerSm;
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, block.x*block.y*block.z, 0));
#if DEBUG_MODE
    printf("processors: %d\nblocks per SM: %d\nknot points:%d\n", numProcs, numBlocksPerSm, KNOT_POINTS);
#endif /* #if DEBUG_MODE */
    if(KNOT_POINTS > numProcs*numBlocksPerSm){
        //printf("Too many KNOT_POINTS. Device supports [%d] blocks, [%d] SMs. Use the new algo\n",numProcs*numBlocksPerSm, numProcs);
        return numProcs*numBlocksPerSm;
    }
    else{
        //printf("Sufficient blocks for given KNOT_POINTS. Device supports [%d] blocks, [%d] SMs. Use the old algo\n",numProcs*numBlocksPerSm, numProcs);
        return KNOT_POINTS;
    }
}

#endif /* GATO_UTILS_CUH */