#include "product/product.h"
#include "matrix/formats/matrixEllpack.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

const int BD = 32;
const dim3 BLOCK_DIM(BD,BD);


__global__ void gpuMatrixMultiVectorELL(int rowsA, int colsA, int colsMulti, const double* A_values ,const int* A_cols, const double* multiVect, double* y) {

    //Indici del thread
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //Indici del blocco.
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Individuo inizio e fine della sottomatrice da utilizzare.
    int aBegin = colsA * BD * by;
    int aEnd   = aBegin + colsA-1 ;
    int aStep  = BD;

    // Individuo inizio e fine della sottomatrice da utilizzare
    int bBegin = BD* bx;
    int bStep  = BD* colsMulti;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    double Csub = 0;

    if(tx<colsA && ty<rowsA){
        for (int a = aBegin, b = bBegin;a <= aEnd;a += aStep, b += bStep) {
            // Declaration of the shared memory array As used to
            // store the sub-matrix of A
            __shared__ double As[BD][BD];

            // Declaration of the shared memory array Bs used to
            // store the sub-matrix of B
            __shared__ double Bs[BD][BD];

            int pos = a + colsA * ty + tx;
            if(pos<rowsA*colsA){
                int posCols = b + colsMulti * A_cols[pos] + tx;
                
                // Load the matrices from device memory
                // to shared memory; each thread loads
                // one element of each 
                As[ty][tx] = A_values[pos];

                if(posCols<colsMulti*rowsA)
                    Bs[ty][tx] = multiVect[posCols];
                else
                    return;
            }
            else 
                return;

            
            
            // Synchronize to make sure the matrices are loaded
            __syncthreads();


            // Multiply the two matrices together;
            // each thread computes one element
            // of the block sub-matrix
            
            if(tx>=colsMulti)
                return;

            for (int k = 0; k < BD; k++) {
                Csub +=  As[ty][k] * Bs[k][tx];
            }


        }
            

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();


        int posC = colsMulti * BD * by + BD * bx+ colsMulti * ty + tx;
        if(posC<rowsA*colsMulti)
            y[posC] = Csub;

        
    }

}




/**
 * Converte una matrice di double in un array di double
 */
void convert2Dto1DDouble(double** mat, double* vet,int rows,int cols){
     for(int i=0,k=0; i<rows; i++)
        for(int j=0; j<cols;j++, k++){
            vet[k] = mat[i][j];
        }
}

/**
 * Converte una matrice di inter in un array di inter
 */
void convert2Dto1DInt(int** mat, int* vet,int rows,int cols){
     for(int i=0,k=0; i<rows; i++)
        for(int j=0; j<cols;j++, k++)
            vet[k] = mat[i][j];
}

/**
 * Converte un vettore in una matrice in formato Matrix. 
 * TODO: potrebbe avere senso implementare questa funzione direttamente in Matrix
 */
void convertToMatrixFormat(double* h_y, Matrix *mResult){
    for(int i=0; i<mResult->rows ; i++)
        for(int j=0; j<mResult->cols ; j++){
            mResult->put(mResult,i,j,h_y[i*mResult->cols+ j]);
        }    
}

/**
 * Funzione definita in product.h che invoca il kernel sulla GPU e raccoglie dati sulle prestazioni.
 */
int productMatrixMatrixParallelEllpack(Matrix *matrix1, Matrix *matrix2, Matrix *mResult, Sample *sample){
    
    DataEllpack * dataEllpack = (DataEllpack*) matrix1->data;
    double** multiVector = (double**) matrix2->data;

    //Strutture usate nella misurazione delle prestazioni
    struct timespec  tStart;
    struct timespec  tEnd;

    int dimMatrix = dataEllpack->colsSubMat * dataEllpack->rowsSubMat;
    int dimMulti  = matrix2->cols * matrix2->rows;
    int dimResult = dataEllpack->rowsSubMat * matrix2->cols;

    // ---------------------- Host memory initialisation ---------------------- //

    double  *h_A_values   = (double *) calloc(dimMatrix, sizeof(double));
    double  *h_Multi_Vec  = (double *) calloc(dimMulti , sizeof(double));
    double  *h_y          = (double *) calloc(dimResult, sizeof(double));
    int     *h_A_cols     = (int *)    calloc(dimMatrix, sizeof(int));



    convert2Dto1DDouble(dataEllpack->matValues,h_A_values  ,dataEllpack->rowsSubMat, dataEllpack->colsSubMat);
    convert2Dto1DDouble(multiVector           ,h_Multi_Vec ,matrix2->rows, matrix2->cols);
    convert2Dto1DInt   (dataEllpack->matCols  ,h_A_cols    ,dataEllpack->rowsSubMat, dataEllpack->colsSubMat);


    // ---------------------- Device memory initialisation ---------------------- //

    double  *d_A_values;
    double  *d_Multi_Vec;
    double  *d_y;
    int     *d_A_cols;


    checkCudaErrors(cudaMalloc((void**) &d_A_values , dimMatrix*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_Multi_Vec, dimMulti *sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_y        , dimResult*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_A_cols   , dimMatrix*sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_A_values   , h_A_values,   dimMatrix * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Multi_Vec  , h_Multi_Vec,  dimMulti  * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_A_cols     , h_A_cols,     dimMatrix * sizeof(int)   , cudaMemcpyHostToDevice));


    // ---------------------- GPU ---------------------- //

    dim3 GRID_DIM((matrix2->cols + BLOCK_DIM.x)/ BLOCK_DIM.x, (dataEllpack->rowsSubMat + BLOCK_DIM.y)/ BLOCK_DIM.y);
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    cudaStream_t stream;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    checkCudaErrors(cudaEventRecord(start));
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaEventRecord(start, stream));

    gpuMatrixMultiVectorELL<<<GRID_DIM, BLOCK_DIM,0,stream >>>(dataEllpack->rowsSubMat, dataEllpack->colsSubMat,matrix2->cols ,d_A_values, d_A_cols, d_Multi_Vec,d_y);

    
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float time = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));

    // ---------------------- Collect result ---------------------- //

    mResult->rows= dataEllpack->rowsSubMat;
    mResult->cols= matrix2->cols;
    
    cudaMemcpy(h_y, d_y, dimResult* sizeof(double), cudaMemcpyDeviceToHost);
    convertToMatrixFormat(h_y, mResult); 

    sample->execTimeSecs = 0;
    sample->execTimeNsecs = time*1000000;
    sample->productName = (char *)__func__;

    cudaFree(d_y);
    cudaFree(d_A_values);
    cudaFree(d_Multi_Vec);
    cudaFree(d_A_cols);

    return 0;
}


