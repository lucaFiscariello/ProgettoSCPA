#include <stdio.h>
#include <malloc.h>
#include "matrix/formats/multiVector.h"
#include "logger/logger.h"


/**
 * Funzione che permette di salvare un valore in una specifica riga o colonna della matrice.
*/
int putMultiVector(Matrix *self, int r, int c, double val){
    
    double** data = (double**)self->data;
    data[r][c] = val;

    return 0;
 
}


/**
 * Funzione che permette di leggere un valore in una specifica riga o colonna della matrice.
*/
double getMultiVector(Matrix *self, int r, int c){
    
    double** data = (double**)self->data;
    return data[r][c];

}

/**
 * Funzione che permette di stampare tutti i non zero della matrice
*/
void printMultiVector(Matrix *self){

    for(int i=0; i<self->rows; i++)
        for(int j=0; j<self->cols; j++){
            logMsg(LOG_TAG_I, "MultiVettore[%d][%d] =  %f\n",i,j,self->get(self,i,j));
        }
    
}

long getSizeMultiVector(Matrix *self){
    return self->rows*self->cols*sizeof(double);
}

/**
 * Funzione per deallocare matrice in memoria
*/
void freeMultiVector(Matrix *self){
    free(self->data);
    free(self);
}

/**
 * Costruttore del multivettore
*/
Matrix* newMultiVector(int rows, int cols) {

    Matrix* matrix = newMatrix();

    //Inizializzo multivettore 
    double ** dataMultiVetor = calloc( rows,sizeof(double* ));
    for(int i=0; i<rows; i++){
        dataMultiVetor[i] = calloc(cols, sizeof(double));
    }

    //Inizializzo matrice da resitituire
    matrix->data = dataMultiVetor;
    matrix->put = putMultiVector;
    matrix->get = getMultiVector;
    matrix->print = printMultiVector;
    matrix->getSize = getSizeMultiVector;
    matrix->cols = cols;
    matrix->rows = rows;
    matrix->numNonZero = cols*rows;

    return matrix;
}
