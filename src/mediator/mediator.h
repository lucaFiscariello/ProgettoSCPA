#ifndef MEDIATOR_H
#define MEDIATOR_H

#include "matrix/matrix.h"
#include "matrix/formats/mm/mm.h"


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
 * Converte una matrice MatrixMarket in un formato specifico passato in input.
 * È sensibilmente più veloce delle altre varianti di convert(), ma funziona solo
 * con matrici in formato MatrixMarket.
*/
void convertFromMM(DataMM *dataMM, Matrix *matrixTo);

#endif 