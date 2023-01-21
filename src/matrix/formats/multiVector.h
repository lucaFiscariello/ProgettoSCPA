#ifndef MULTIVECTOR_H
#define MULTIVECTOR_H

#include "matrix/matrix.h"

/**
 * Costruttore del multivettore
*/
Matrix* newMultiVector(int rows, int cols);

/**
 * Funzione per deallocare matrice in memoria
*/
void freeMultiVector(Matrix *self);




#endif // MULTIVECTOR_H