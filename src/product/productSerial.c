#include "product/product.h"

double** productMatrixMatrixSerial(Matrix* matrix1, Matrix* matrix2){

    NotZeroElement* nze;
    int finalRows = matrix1->rows;
    int finalCols = matrix2->cols;

    //Inizializzo matrice risultato
    double** result = calloc(finalRows, sizeof(double*));
    for(int i =0; i<finalRows;i++){
        result[i] = calloc(finalCols, sizeof(double));
    }

    //Scorro tutti gli elementi non zeri della prima matrice
    for(int i =0; i< matrix1->numNonZero; i++){
        
        nze = matrix1->getNonZero(matrix1,i);

        //scorro tutti gli elementi della riga "nze->row" della seconda matrice
        for(int j =0; j< matrix2->cols; j++){

            /**
             * Moltiplico un elemento non nullo della prima matrice per tutti gli elementi della riga della seconda matrice.
             * Sommo questo risultato parziale nella matrice risultato. La posizione in cui sommare questo risultato parziale
             * è definita dalla riga dell'elemento della prima matrice e dalla colonna del valore della seconda matrice.
             */
            result[nze->row][j] += nze->value * matrix2->get(matrix2,nze->row,j);
        }
    }

    return result;
}
