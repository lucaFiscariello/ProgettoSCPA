#ifndef COO_H
#define COO_H

#include "matrix/matrix.h"

/**
 * The representation of a COO sparse matrix data.
 * Values of same index between lists are of the same matrix element.
*/
typedef struct coo_data{

    Node *elements; /** list of non-null matrix elements*/
    Node *cols;  /** list of col indexes.*/
    Node *rows; /** list of row indexes*/

} DataCOO;

/** constructor*/
Matrix *newMatrixCOO();

/** destructor*/
void freeMatrixCOO(Matrix *self);

#endif // COO_H


