#include "matrix/formats/arrayDense.h"
#include "logger/logger.h"
#include <stdio.h>

int to1D(int rSize, int r, int c){
    return r * rSize + c;
}

int putArrayDense(Matrix *self, int r, int c, double val){

    double *data = (double *) self ->data;
    
    // indexes sanity check
    ON_ERROR_LOG_AND_RETURN(outOfBounds(self, r, c), -1, "Invalid coordinates: r=%d(%d), c=%d(%d)\n", r, self ->rows, c, self ->cols);

    data[to1D(self ->cols, r, c)] = val;
    
    return 0; 
}

double getArrayDense(Matrix *self, int r, int c){

    double *data = (double *) self ->data;
    
    // indexes sanity check
    ON_ERROR_LOG_AND_RETURN(outOfBounds(self, r, c), -1, "Invalid coordinates: r=%d(%d), c=%d(%d)\n", r, self ->rows, c, self ->cols);

    return data[to1D(self ->cols, r, c)]; 
}

void printArrayDense(Matrix *self){

    double *data = (double *) self ->data;

    for (int r = 0; r < self ->rows; r++){
        printf("[ ");
        for (int c = 0; c < self ->cols; c++){
            printf("%f ", data[to1D(self ->cols, r, c)]);
        }
        printf("]\n");
    }
}

Matrix *newArrayDenseMatrix(int rows, int cols){

    Matrix *self = newMatrix();
    self ->data = calloc(rows * cols, sizeof(double));

    self ->rows = rows;
    self ->cols = cols;

    // since it is a dense matrix we don't implement sparse matrix utilities
    self ->numNonZero = -1;

    self ->put = putArrayDense;
    self ->get = getArrayDense;
    self ->print = printArrayDense;

    return self; 
}

void freeArrayDenseMatrix(Matrix *self){
    free(self ->data);
    free(self);
}