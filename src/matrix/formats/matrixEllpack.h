#ifndef MATRIXELLPACK_H
#define MATRIXELLPACK_H

#include "matrix/matrix.h"

/**
 * Costruttore della matrice in formato ellpack
*/
Matrix* newMatrixEllpack(int cols, int rows);

/**
 *  Contiene le strutture dati utili per gestire in memoria una matrice sparsa in formato Ellpack.
*/
typedef struct DataEllpack{

    /**
     * Valori della matrice sparsa memorizzati in una sottomatrice.
    */
    double ** matValues;

    /**
     * Un elemento di questa matrice contiene la colonna di appartenenza del valore omologo memorizzato in matValues
    */
    int** matCols;

    /**
     * Numero di colonne che ha la  matrice sparsa.
    */
    int colsSubMat;

    /**
     * Vettore che ha tante componenti quante le righe della matrice sparsa. Serve per capire per ogni riga dove 
     * posso scrivere un nuovo valore.
    */
    int* nextInsert ;

}DataEllpack;

#endif // MATRIXELLPACK_H