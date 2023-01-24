#include "matrix/formats/coo.h"
#include "logger/logger.h"
#include <stdlib.h>
#include <stdio.h>

double getCOO(Matrix *self, int r, int c){

    DataCOO *data = (DataCOO *)self->data;

    for (int i = 0; i < self->numNonZero; i ++){
        if (data ->rows[i] == r && data ->cols[i] == c){
            return data ->elements[i];
        }
    }
    return 0;
}

NotZeroElement* getNonZeroCOO(Matrix *self, int pos){

    NotZeroElement* nze;
    DataCOO *data = (DataCOO *)self->data;    
    
    ON_ERROR_LOG_AND_RETURN((pos < 0 || pos >= self ->numNonZero), NULL, "pos %d is out of bounds\n", pos);
    
    nze = calloc(1, sizeof(NotZeroElement));
    nze ->row = data ->rows[pos];
    nze ->col = data ->cols[pos];
    nze ->value = data ->elements[pos];
    
    return nze;
}

int putCOO(Matrix *self, int r, int c, double val){

    DataCOO *data = (DataCOO *)self->data;
    
    // Ingrandiamo la dimensione della matrice se necessario.
    if (r >= self ->rows){
        self ->rows = r + 1;
    }
    if (c >= self ->cols){
        self ->cols = c + 1;
    }

    // Scorro tutti gli elementi della matrice cercando la coppia di indici (r,c)
    for (int i = 0; i < self ->numNonZero; i ++){
        
        if (data ->rows[i] == r && data ->cols[i] == c){
                        
            // la riga e la colonna specificati sono già occupati da un elemento non zero, che andremo
            // a sostituire con il nuovo valore
            if (val != 0){
                data ->elements[i] = val;
                return 0;
            } else {
                // val è zero. Per mantenere la lista senza elementi nulli, prendo l'elemento
                // non-zero alla fine dell'array e lo metto al suo posto, aggiornando gli indici.
                if (self ->numNonZero - 1 > 0){
                    data ->elements[i] = data ->elements[self ->numNonZero - 1];
                    data ->rows[i] = data ->rows[self ->numNonZero - 1];
                    data ->cols[i] = data ->cols[self ->numNonZero - 1];
                } else {
                    // era rimasto un solo elemento non-zero, la sua eliminazione
                    // comporta il reset della capacità degli array
                    free(data ->elements);
                    free(data ->rows);
                    free(data ->cols);
                    data ->elements = NULL;
                    data ->rows = NULL;
                    data ->cols = NULL;
                    data ->capacity = 0;
                }
                self ->numNonZero --;
            }
        }
    }

    // Devo aggiungere elemento non-zero poiché non è stato trovato tra quelli già
    // presenti nella matrice
    if (val != 0){
        
        // gonfiamo arrays se non c'è spazio per l'elemento
        if (self ->numNonZero + 1 > data ->capacity){
            data ->elements = reallocarray(data ->elements, data ->capacity + 1, sizeof(double));
            data ->rows = reallocarray(data ->rows, data ->capacity + 1, sizeof(int));
            data ->cols = reallocarray(data ->cols, data ->capacity + 1, sizeof(int));
            data ->capacity ++;
        }
        data ->elements[self ->numNonZero] = val;
        data ->rows[self ->numNonZero] = r;
        data ->cols[self ->numNonZero] = c;
        self ->numNonZero ++;  
    }

    return 0;
}

void printCOO(Matrix *self){

    DataCOO *data = (DataCOO *)self->data;
    
    printf("%-6s, %-6s, %-6s\n", "row", "col", "value");
    for (int i = 0; i < self ->numNonZero; i++)
    {
        printf("%-6d, %-6d, %-6.6f\n", data ->rows[i], data ->cols[i], data ->elements[i]);
    }
}

Matrix *newMatrixCOO(){

    Matrix *self = newMatrix();
    DataCOO *data = calloc(1, sizeof(DataCOO));
    
    /** setup members*/
    self ->data = data;
    
    /** setup methods*/
    self ->put = putCOO;
    self ->get = getCOO;
    self ->getNonZero = getNonZeroCOO;
    self ->print = printCOO;

    return self;
}

void freeMatrixCOO(Matrix *self){
    
    DataCOO *data = (DataCOO *)self->data;

    // free data
    free(data ->elements);
    free(data ->rows);
    free(data ->cols);
    free(data);

    // free matrix
    freeMatrix(self);
}

