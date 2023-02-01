#include "product/product.h"
#include "matrix/formats/matrixEllpack.h"
#include "matrix/formats/multiVector.h"
#include "matrix/formats/arrayDense.h"
#include "mediator/mediator.h"
#include <omp.h>
#include <time.h>

int productEllpackMultivectorParallelCPU(Matrix *matrix1, Matrix *matrix2, Matrix *mResult, Sample *sample){

    DataEllpack *ellpackData = (DataEllpack *) matrix1->data;
    double **multivectorData = (double **) matrix2->data;

    struct timespec start, end;
    double numMBytesEllpack, numMBytesMultivector;

    // we store the result in a buffer array in order to not taint perfomance
    // measurements with the time needed to write the result in the result matrix
    Matrix *result = newArrayDenseMatrix(matrix1 ->rows, matrix2->cols);
    double *resultData = (double *) result ->data;

    // TODO: start measuring
    sample ->productName = (char *)__func__;
    clock_gettime(CLOCK_MONOTONIC, &start);

        
    /**
     * All variables used in the parallel region are shared unless specified otherwise.
     * 
     * schedule(static):
     *  Each thread will be assigned a fixed number of iterations.
     *  Since each iteration should do the same amount of work, this should be a faster option
     *  compared to a dynamic schedule, since the implies synchronization overhead.
     * 
     * collapse(3):
     *  All for cycles are collapsed into a single loop, so its iterations
     *  are distributed among the threads. This is possibile because each loop has a
     *  fixed number of iterations indipendent from other loops.
    */
    #pragma omp parallel for default(shared) schedule(static) collapse(3) 
    for (int r1 = 0; r1 < ellpackData ->rowsSubMat; r1++){
        for (int c2 = 0; c2 < matrix2 -> cols; c2 ++){
            for (int cSub = 0; cSub < ellpackData ->colsSubMat; cSub++){
                                    
                /**
                 * Ogni elemento della matrice risultato è aggiornato atomicamente, implicando
                 * una sincronizzazione tra i thread dedicati a uno stesso elemento, ma non tra quelli
                 * dedicati a elementi diversi.
                 * https://www.openmp.org/spec-html/5.0/openmpsu95.html#x126-4840002.17.7
                */
                #pragma omp atomic
                resultData[r1 * matrix2 ->cols + c2] += ellpackData ->matValues[r1][cSub] * multivectorData[ellpackData ->matCols[r1][cSub]][c2];
            }
        }
    }
    
    // TODO: stop measuring and write the measurements down in the sample
    clock_gettime(CLOCK_MONOTONIC, &end);
    sample ->execTimeSecs = end.tv_sec - start.tv_sec;
    sample ->execTimeNsecs = end.tv_nsec - start.tv_nsec;
    sample ->gflops = calcGflops(sample ->execTimeSecs, sample ->execTimeNsecs, matrix1 ->numNonZero, matrix2 ->cols);
    numMBytesEllpack = (sizeof(double) * matrix1 ->numNonZero) / 1e6;
    numMBytesMultivector = sizeof(double) * matrix2 ->rows * matrix2 ->cols / 1e6;
    double size = numMBytesEllpack + numMBytesMultivector;
    sample ->bandwidth = calcBandwidth(size, sample ->execTimeSecs, sample ->execTimeNsecs);

    // write the result in the result matrix
    convert_dense_too(result, mResult);

    return 0;
}
