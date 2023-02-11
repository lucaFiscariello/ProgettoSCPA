#include "matrix/matrix.h"
#include <stdlib.h>
#include "logger/logger.h"

int putMatrix(Matrix *self, int r, int c, double val){
    LOG_UNIMPLEMENTED_CALL();
    return 0;
}

double getMatrix(Matrix *self, int r, int c){
    LOG_UNIMPLEMENTED_CALL();
    return 0;
}

void printMatrix(Matrix *self){
    LOG_UNIMPLEMENTED_CALL();
}

long getSizeMatrix(Matrix *self){
    LOG_UNIMPLEMENTED_CALL();
    return 0;
}

Matrix *cloneEmptyMatrix(Matrix *self){
    LOG_UNIMPLEMENTED_CALL();
    return NULL;
}

Matrix *newMatrix(){
    Matrix *self = calloc(1, sizeof(Matrix));
    self ->formatName = "Unimplemented";

    self ->put = putMatrix;
    self ->get = getMatrix;
    self ->print = printMatrix;
    self ->getSize = getSizeMatrix;
    self ->cloneEmpty = cloneEmptyMatrix;

}

void freeMatrix(Matrix *self){
    free(self);
}

bool outOfBounds(Matrix *self, int r, int c){
    return (r < 0 || r >= self ->rows || c < 0 || c >= self ->cols);
}