#include "product/product.h"
#include "matrix/formats/matrixEllpack.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

#define BD 64

const dim3 BLOCK_DIM(BD);
const dim3 GRID_DIM(1);

/**
 * Per come è implementato attualmente il kernel ogni thread si prende una riga della prima matrice e la moltuplica per tutte le colonne
 * della seconda matrice. Si parte da questa versione base e si introducono le varie ottimizzazioni.
*/
__global__ void gpuMatrixMultiVectorELL(int rows, int cols, int colsMulti, const double* A_values ,const int* A_cols, const double* multiVect, double* y) {
    
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if(idx < rows){
        for(int colMulti=0;colMulti<colsMulti;colMulti++)
            for(int colMat=0;colMat<cols;colMat++)
                y[idx * colsMulti +colMulti] += A_values[idx*cols+ colMat] * multiVect[A_cols[idx*cols+colMat]*colsMulti+colMulti];       
    }

}

/**
 * Converte una matrice di double in un array di double
 */
void convert2Dto1DDouble(double** mat, double* vet,int rows,int cols){
     for(int i=0,k=0; i<rows; i++)
        for(int j=0; j<cols;j++, k++)
            vet[k] = mat[i][j];
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

    int dimMatrix = matrix1->cols * matrix1->rows;
    int dimMulti  = matrix2->cols * matrix2->rows;
    int dimResult = matrix1->rows * matrix2->cols;

    // ---------------------- Host memory initialisation ---------------------- //

    double  *h_A_values   = (double *) calloc(dimMatrix, sizeof(double));
    double  *h_Multi_Vec  = (double *) calloc(dimMulti , sizeof(double));
    double  *h_y          = (double *) calloc(dimResult, sizeof(double));
    int     *h_A_cols     = (int *)    calloc(dimMatrix, sizeof(int));


    convert2Dto1DDouble(dataEllpack->matValues,h_A_values  ,matrix1->rows, matrix1->cols);
    convert2Dto1DDouble(multiVector           ,h_Multi_Vec ,matrix2->rows, matrix2->cols);
    convert2Dto1DInt   (dataEllpack->matCols  ,h_A_cols    ,matrix1->rows, matrix1->cols);


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
    checkCudaErrors(cudaMemcpy(d_y          , h_y,          dimResult * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_A_cols     , h_A_cols,     dimMatrix * sizeof(int)   , cudaMemcpyHostToDevice));


    // ---------------------- GPU ---------------------- //

    clock_gettime(CLOCK_REALTIME,&tStart);
    gpuMatrixMultiVectorELL<<<GRID_DIM, BLOCK_DIM >>>(matrix1->rows, matrix1->cols,matrix2->cols ,d_A_values, d_A_cols, d_Multi_Vec,d_y);
    clock_gettime(CLOCK_REALTIME ,&tEnd);      

    checkCudaErrors(cudaDeviceSynchronize());


    // ---------------------- Collect result ---------------------- //

    cudaMemcpy(h_y, d_y, dimResult* sizeof(double), cudaMemcpyDeviceToHost);
    convertToMatrixFormat(h_y, mResult); 

    sample->execTimeSecs = tEnd.tv_sec - tStart.tv_sec;
    sample->execTimeNsecs = tEnd.tv_nsec - tStart.tv_nsec;
    sample->gflops= (double)2*matrix2->cols*matrix1->numNonZero/sample->execTimeNsecs;
    sample->bandwidth = (double) sizeof(double)*(matrix1->numNonZero + matrix2->cols * matrix2->rows)/sample->execTimeNsecs;
    sample->productName = (char*)calloc(strlen(__func__),sizeof(char));
    strcpy(sample->productName, __func__);  

    return 0;
}


