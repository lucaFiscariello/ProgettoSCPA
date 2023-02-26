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

    // Csub è l'elemento della sottomatrice calcolato da un thread
    double Csub = 0;

    
    //Definizione della memoria condivisa. Ogni blocco di thread ha associato un blocco di memoria ocndivisa di BD*BD
     __shared__ double As[BD][BD];
     __shared__ double Bs[BD][BD];

    int pos;

    if(tx<colsA && ty<rowsA && tx<colsMulti){
        for (int a = aBegin, b = bBegin;a <= aEnd;a += aStep, b += bStep) {
            

            pos = a + colsA * ty + tx;
            if(pos<rowsA*colsA){
                int posCols = b + colsMulti * A_cols[pos] + tx;
                
               //Carico i dati in memoria condivisa escludendo il padding 
                As[ty][tx] = A_values[pos];
                if(posCols<colsMulti*rowsA)
                    Bs[ty][tx] = multiVect[posCols];
                else
                    return;
                
            }
            else 
                return;

            
            
            //Sincornizzo i threads per essere sicuro che tutti abbiano scritto in memoria condivisa
            __syncthreads();


            //Calcolo il prodotto
            for (int k = 0; k < BD; k++) {
                Csub +=  As[ty][k] * Bs[k][tx];
            }


        }
            

        // Sincornizzo dopo aver terminato la moltiplicazione, in attesa che tutti i thread abbiano completato i propri calcoli
        __syncthreads();


        //Salvo risultato
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


