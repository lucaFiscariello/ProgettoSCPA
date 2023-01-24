#ifndef COO_H
#define COO_H

#include "matrix/matrix.h"

/**
 * The representation of a COO sparse matrix data.
 * Values of same index between arrays are of the same matrix element.
*/
typedef struct coo_data{

    double *elements; /** array of non-null matrix elements*/
    int *cols;  /** array of col indexes.*/
    int *rows; /** array of row indexes*/
    int capacity; /** actual size of arrays*/

} DataCOO;

/** constructor*/
Matrix *newMatrixCOO();

/** destructor*/
void freeMatrixCOO(Matrix *self);

#endif // COO_H


