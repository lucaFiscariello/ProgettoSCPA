#include <stdio.h>
#include "header/matrixEllpack.h"
#include <malloc.h>
#include "header/logger.h"


/**
 * Funzione che permette di salvare un valore in una specifica riga o colonna della matrice.
*/
void putEllpack(Matrix *self, int r, int c, double val){
    
    DataEllpack* data = (struct DataEllpack*)self->data;

    //Controllo se ho spazio nella riga in cui voglio scrivere un nuovo valore.
    if(data->nextInsert[r] == data->colsSubMat){

        //Se ho riempito tutta la riga rialloco le matrici aggiungendo una nuova colonna
        for(int i=0; i<self->rows; i++){
            data->matValues[i] = (double *) realloc(data->matValues[i],sizeof(data->matValues[i]) + sizeof(double));
            data->matCols[i] = (int *) realloc(data->matValues[i],sizeof(data->matValues[i])+sizeof(int));
        }
    }


    //Salvo il valore modificando opportunamente le due matrici
    data->matValues[r][data->nextInsert[r]] = val ;
    data->matCols[r][data->nextInsert[r]] = c ;

    //Aggiorno indici per le successive scritture
    data->nextInsert[r]++;

    //incremento numeri non zero
    self->numNonZero++;


}


/**
 * Funzione che permette di leggere un valore in una specifica riga o colonna della matrice.
*/
double getEllpack(Matrix *self, int r, int c){
    
    DataEllpack* data = (struct DataEllpack*)self->data;
    int columnFound = -1;

    //Scorro la riga r della matrice contentente gli indici di colonna della matrice sparsa
    for(int j=0; j< data->colsSubMat; j++){
        columnFound++;

        //verifico se la matrice con gli indici di colonna contiene l'indice di colonna c che sto cercando
        if(data->matCols[r][j] == c)
           break;
    }

    //se nella riga r non trovo l'indice di colonna c allora in quella posizione nella matrice sparsa ci sarà uno 0
    if(columnFound==data->colsSubMat)
        return 0;
    
    return data->matValues[r][columnFound];

}


/**
 * Costruttore della matrice in formato ellpack
*/
Matrix* newMatrixEllpack(int cols, int rows) {

    struct Matrix* matrix =( struct Matrix *) malloc( sizeof( struct Matrix ));
    struct DataEllpack* dataEllpack =( struct DataEllpack *) malloc( sizeof( struct DataEllpack ));

    dataEllpack->colsSubMat=1; // Inizializzo a 1 il numero di colonne della matrice sparsa
    dataEllpack->nextInsert = (int *) malloc(sizeof(int) * rows); 
    
    dataEllpack->matValues = (double **) malloc(sizeof(double*)* rows);
    dataEllpack->matCols = (int **) malloc(sizeof(int*)* rows);

    for(int i=0; i<rows; i++){
        dataEllpack->nextInsert[i] = 0; // tutti i nuovi valori verranno inseriti nella colonna 0
        dataEllpack->matValues[i] = (double *) malloc(sizeof(double));
        dataEllpack->matCols[i] = (int *) malloc(sizeof(int));
    }

    //Inizializzo matrice da resitituire
    matrix->data = dataEllpack;
    matrix->put = putEllpack;
    matrix->get = getEllpack;
    matrix->cols = cols;
    matrix->rows = rows;
    matrix->numNonZero = 0;

    return matrix;
}
