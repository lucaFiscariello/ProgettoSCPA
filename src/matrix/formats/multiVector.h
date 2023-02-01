#ifndef MULTIVECTOR_H
#define MULTIVECTOR_H

#include "matrix/matrix.h"

/**
 * Costruttore del multivettore.
 * Nel campo data della matrice viene salvata una matrice di double (double **).
*/
Matrix* newMultiVector(int rows, int cols);

/**
 * Funzione per deallocare matrice in memoria
*/
void freeMultiVector(Matrix *self);




#endif // MULTIVECTOR_H