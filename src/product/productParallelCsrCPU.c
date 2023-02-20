#include "product/product.h"
#include "matrix/formats/csr.h"
#include "matrix/formats/multiVector.h"
#include "matrix/formats/arrayDense.h"
#include "mediator/mediator.h"
#include <omp.h>
#include <time.h>

int productCsrMultivectorParallelCPU(Matrix *matrix1, Matrix *matrix2, Matrix *mResult, Sample *sample){

    CSRData *data = (CSRData *)matrix1->data;
    double **multivectorData = (double **) matrix2->data;

    int nrows = matrix1->rows;
    int ncols = matrix2->cols;

    struct timespec start, end;

    // we store the result in a buffer array in order to not taint perfomance
    // measurements with the time needed to write the result in the result matrix
    Matrix *result = newArrayDenseMatrix(matrix1 ->rows, matrix2->cols);
    double *resultData = (double *) result ->data;

    sample ->productName = (char *)__func__;
    clock_gettime(CLOCK_MONOTONIC, &start);
        
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            double sum = 0;
            int row_start = data->firstColOfRowIndexes[i];
            int row_end = data->firstColOfRowIndexes[i+1];

            #pragma omp simd reduction(+:sum)
            for (int k = row_start; k < row_end; k++) {
                sum += data->values[k] * multivectorData[data->columns[k]][j];
            }
            resultData[i*ncols+j]=sum;
        }
    }

    logMsg(LOG_TAG_D, "calculations completed\n");
    
    // stop measuring and write the measurements down in the sample
    clock_gettime(CLOCK_MONOTONIC, &end);
    sample ->execTimeSecs = end.tv_sec - start.tv_sec;
    sample ->execTimeNsecs = end.tv_nsec - start.tv_nsec;
    logMsg(LOG_TAG_D, "elapsed time registered in sample\n");

    // write the result in the result matrix
    convert_dense_too(result, mResult);
    freeArrayDenseMatrix(result);
    logMsg(LOG_TAG_D, "conversion completed\n");

    return 0;
}
