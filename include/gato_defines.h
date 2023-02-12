#ifndef GATO_DEFINES_H
#define GATO_DEFINES_H





/* these must be set for compile */
#define BLOCK_J_PRECON  1
#define SS_PRECON       1
#define PRECONDITIONER_BANDWIDTH 3



#define GATO_BLOCK_ID (blockIdx.x)
#define GATO_THREAD_ID (threadIdx.x)
#define GATO_THREADS_PER_BLOCK (blockDim.x)
#define GATO_NUM_BLOCKS   (gridDim.x)
#define GATO_LEAD_THREAD (GATO_THREAD_ID == 0)
#define GATO_LEAD_BLOCK (GATO_BLOCK_ID == 0)
#define GATO_LAST_BLOCK (GATO_BLOCK_ID == GATO_NUM_BLOCKS - 1)
#define DEFAULT_STREAM  0

#define GATO_TIMING 1
#define DEBUG_MODE   0
#define GATO_PRINTING  DEBUG_MODE || 0   
#define PRINT_THREAD    0
#define GATO_PRINT_BLOCK_THREAD (1 && (GATO_BLOCK_ID == 0) && GATO_LEAD_THREAD)
#define GATO_PRINT_BLOCK_THREAD2 (1 && (GATO_BLOCK_ID == 2) && threadIdx.x == 0)


#define STATES_SQ       (STATE_SIZE*STATE_SIZE)
#define CONTROLS_SQ     (CONTROL_SIZE*CONTROL_SIZE)
#define STATES_S_CONTROLS (STATE_SIZE+CONTROL_SIZE)
#define STATES_P_CONTROLS (STATE_SIZE*CONTROL_SIZE)
#define KKT_G_DENSE_SIZE_BYTES   (((STATES_SQ+CONTROLS_SQ)*KNOT_POINTS-CONTROLS_SQ)*sizeof(float))
#define KKT_C_DENSE_SIZE_BYTES   (((STATES_SQ+STATES_P_CONTROLS)*(KNOT_POINTS-1))*sizeof(float))




inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


#endif /* GATO_DEFINES_H */
