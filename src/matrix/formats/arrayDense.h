#ifndef ARRAYDENSE_H
#define ARRAYDENSE_H

#include "matrix/matrix.h"

/**
 * Allocates a matrix with its data stored in a zero-initialiazed 1D double array of
 * fixed length = rows * cols. 
*/
Matrix *newArrayDenseMatrix(int rows, int cols);

void freeArrayDenseMatrix(Matrix *m);

#endif // ARRAYDENSE_H