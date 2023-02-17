#ifndef MEDIATOR_H
#define MEDIATOR_H

#include "matrix/matrix.h"


/**
 * Funzione che permette di convertire una matrice da un formato ad un altro.
 * Funziona solo se le matrici implementano il conteggio dei non-zero e il
 * metodo getNonZero() (matrici sparse).
*/
void convert(Matrix *from, Matrix *to);

/**
 * Converte una matrice da un formato a un altro. Gestisce anche le matrici dense,
 * senza che esse implementino il conteggio dei non-zero e il metodo getNonZero().
 * Potrebbe essere più lenta di convert() per le matrici sparse.
*/
void convert_dense_too(Matrix *from, Matrix *to);


/**
 * Converte una matrice memorizzata in un file in un formato specifico passato in input.
*/
void convertFromFile(const char *filename, Matrix *matrixTo);

#endif 