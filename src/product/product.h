#ifndef PRODUCT_H
#define PRODUCT_H

#include "matrix/matrix.h"


/**
 * Funzione che implementa il prodotto matriciale in modo seriale. La funzione non è tenuta a conoscere il formato 
 * delle due matrici. L'unico vincolo ovviamente è che il numero di righe della prima matrice si 
 * uguale al numero di colonne della seconda matrice.
*/
double** productMatrixMatrixSerial(Matrix* matrix1, Matrix* matrix2);

#endif