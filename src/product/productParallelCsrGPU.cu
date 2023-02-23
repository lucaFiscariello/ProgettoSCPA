#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include "product/product.h"
#include "matrix/matrix.h"
#include "matrix/formats/csr.h"
#include "matrix/formats/arrayDense.h"
#include "mediator/mediator.h"
#include <math.h>
#include "logger/logger.h"
#include <stdint.h>

const int WARP_SIZE = 32;
const int BD = WARP_SIZE;
const dim3 BLOCK_DIM(BD,BD);

/**
 * Used to access memory allocated with cudaMallocPitch.
*/
#define getPitched(base, i, j, pitch, type) *((type *)((uint8_t *)base + i * pitch) + j)

/**
 * Adjusts size to be a multiple of BD more or equal than given size.
 * This should improve perfomance since we don't need to sanitize indexes anymore,
 * thus avoiding divergence.
*/
size_t adjustToWarpSize(size_t size){
    return size + size % WARP_SIZE;
}
/**
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#restrict
*/
__global__ void csrMultivectorKernel(int numSparseRows, int numNonZero,
 int *__restrict__ firstColOfRowIndexes, int *__restrict__ columns, double * __restrict__ values,
 double * __restrict__ mv, size_t mv_pitch, int mvCols, double * __restrict__ result, size_t result_pitch)
{
    int row, startCol, endCol;
    int mvCol;
    int partial = 0;

    /**
     * Associamo ogni thread di un blocco a una riga della matrice.
     * Dunque dobbiamo dividere le righe per ogni blocco a gruppi di BD righe.
     * Ogni thread prende le colonne della riga che gli è stata assegnata e accumula il 
     * prodotto sommando il risultato parziale  al prodotto tra il valore corrispondente alla
     * colonna e il valore del multivettore corrispondente.
    */
    
    row = threadIdx.x + blockIdx.x * blockDim.x;
    mvCol = threadIdx.y + blockIdx.y * blockDim.y;

    startCol = firstColOfRowIndexes[row];
    endCol = firstColOfRowIndexes[row + 1];
    for (int c = startCol; c < endCol; c++){
        partial += values[c] * getPitched(mv, columns[c], mvCol, mv_pitch, double);
    }
    getPitched(result, row, mvCol, result_pitch, double) = partial;
}

int productCsrMultivectorParallelGPU(Matrix *matrix1, Matrix *matrix2, Matrix *mResult,
 Sample *sample)
 {   
    
    // host data
    CSRData *csrData = (CSRData *)matrix1->data;
    Matrix *result;
    Matrix *multiVector1D = newArrayDenseMatrix(matrix2 ->rows, matrix2 ->cols);
    double *h_result;
    
    // device data
    int *d_csr_firstColOfRowIndexes;
    int *d_csr_columns;
    double *d_csr_values;

    double *d_mv;
    size_t d_mv_pitch;    

    double *d_result;
    size_t d_result_pitch;

    // we convert the multivector to a 1D array
    convert(matrix2, multiVector1D);

    // will use events to measure time
    // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    cudaEvent_t start, stop;
    cudaStream_t stream = NULL; // default stream
    float execTimeMsecs;
    size_t d_csr_firstColOfRowIndexes_capacity, d_csr_columns_capacity, d_csr_values_capacity,
     d_mv_width_capacity, d_mv_height_capacity, d_result_width_capacity, d_result_height_capacity;

    sample ->productName = (char *)__func__;

    // ---------------------- Host memory initialisation ---------------------- //

    result = newArrayDenseMatrix(matrix1 ->rows, matrix2 ->cols);
    h_result = (double *)result ->data;

    // ---------------------- GPU ---------------------- //
    
    // prepare device
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaDeviceReset());
    
    // ---------------------- Device memory initialisation ---------------------- //
    
    d_csr_firstColOfRowIndexes_capacity = adjustToWarpSize(sizeof(int) * csrData->numCompressedRows);
    d_csr_columns_capacity = adjustToWarpSize(sizeof(int) * matrix1 -> numNonZero);
    d_csr_values_capacity = adjustToWarpSize(sizeof(double) * matrix1 -> numNonZero);
    d_mv_width_capacity = adjustToWarpSize(matrix2 ->cols * sizeof(double));
    d_mv_height_capacity = adjustToWarpSize(matrix2 -> rows);
    d_result_width_capacity = adjustToWarpSize(matrix2 ->cols * sizeof(double));
    d_result_height_capacity = adjustToWarpSize(matrix1 -> rows);
    
    checkCudaErrors(cudaMalloc((void **) &d_csr_firstColOfRowIndexes, d_csr_firstColOfRowIndexes_capacity));
    checkCudaErrors(cudaMalloc((void **) &d_csr_columns, d_csr_columns_capacity));
    checkCudaErrors(cudaMalloc((void **) &d_csr_values, d_csr_values_capacity));
    checkCudaErrors(cudaMallocPitch((void **) &d_mv, &d_mv_pitch, d_mv_width_capacity, d_mv_height_capacity));
    checkCudaErrors(cudaMallocPitch((void **) &d_result, &d_result_pitch, d_result_width_capacity, d_result_height_capacity));
    
    checkCudaErrors(cudaMemset(d_csr_firstColOfRowIndexes, 0, d_csr_firstColOfRowIndexes_capacity));
    checkCudaErrors(cudaMemset(d_csr_columns, 0, d_csr_columns_capacity));
    checkCudaErrors(cudaMemset(d_csr_values, 0, d_csr_values_capacity));
    checkCudaErrors(cudaMemset2D(d_mv, d_mv_pitch, 0, d_mv_width_capacity, d_mv_height_capacity));
    checkCudaErrors(cudaMemset2D(d_result, d_result_pitch, 0, d_result_width_capacity, d_result_height_capacity));
    
    // we enlarge bank size to make enough room to store at least a double in each bank
    // https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
    checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

    // copy data to device
    checkCudaErrors(cudaMemcpy(d_csr_firstColOfRowIndexes, csrData->firstColOfRowIndexes, sizeof(int) * csrData->numCompressedRows, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csr_columns, csrData->columns, sizeof(int) * matrix1 -> numNonZero, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csr_values, csrData->values, sizeof(double) * matrix1 -> numNonZero, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(d_mv, d_mv_pitch, (double *)multiVector1D ->data, multiVector1D ->cols * sizeof(double), multiVector1D ->cols * sizeof(double), multiVector1D ->rows, cudaMemcpyHostToDevice));

    // initialize timer
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // start timer
    checkCudaErrors(cudaEventRecord(start, stream));

    // launch kernel
    dim3 GRID_DIM(ceil(((double)csrData ->numCompressedRows - 1) / BLOCK_DIM.x), ceil(((double)matrix2 ->cols) / BLOCK_DIM.y));
    csrMultivectorKernel<<<GRID_DIM, BLOCK_DIM>>>(csrData ->numCompressedRows - 1, 
     matrix1 ->numNonZero, d_csr_firstColOfRowIndexes, d_csr_columns, d_csr_values, d_mv,
     d_mv_pitch, matrix2 ->cols, d_result, d_result_pitch);

    // stop timer and calculate elapsed time
    checkCudaErrors(cudaEventRecord(stop, stream));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&execTimeMsecs, start, stop));
    sample ->execTimeNsecs = execTimeMsecs * 1e6;

    // copy result from device to host
    checkCudaErrors(cudaMemcpy2D(h_result, matrix2 ->cols * sizeof(double), d_result, d_result_pitch, matrix2 ->cols * sizeof(double), matrix1 ->rows, cudaMemcpyDeviceToHost));
    convert(result, mResult);

    // clean up
    freeArrayDenseMatrix(result);
    checkCudaErrors(cudaFree(d_csr_firstColOfRowIndexes));
    checkCudaErrors(cudaFree(d_csr_columns));
    checkCudaErrors(cudaFree(d_csr_values));
    checkCudaErrors(cudaFree(d_result));
    checkCudaErrors(cudaFree(d_mv));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop)); 
    freeArrayDenseMatrix(multiVector1D); 

    return 0;
}
