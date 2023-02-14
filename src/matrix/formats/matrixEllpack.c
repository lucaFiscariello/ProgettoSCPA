#include <stdio.h>
#include "matrix/formats/matrixEllpack.h"
#include <malloc.h>
#include "logger/logger.h"


/**
 * Funzione che permette di salvare un valore in una specifica riga o colonna della matrice.
*/
int putEllpack(Matrix *self, int r, int c, double val){
    
    if(val == 0)
        return -1;
        
    DataEllpack* data = (DataEllpack*)self->data;

    //Controllo se ho abbastanza righe nella matrice
    if(data->rowsSubMat <= r){
        
        //Se non ho abbastanza righe rialloco la matrice aggiungendo il numero di righe necesssarie
        data->matValues  = (double **)realloc(data->matValues, sizeof(double*)*(r+1));
        data->matCols    = (int **)realloc(data->matCols,   sizeof(int*)*(r+1));
        data->nextInsert = (int *)realloc(data->nextInsert,sizeof(int)*(r+1));
        
        //Alloco le colonne associate alle righe appena aggiunte e pulisco le nuove caselle di nextInsert
        for(int k=data->rowsSubMat; k<r+1; k++ ){
            data->matValues[k] =  (double *)calloc(data->colsSubMat,sizeof(double));
            data->matCols[k]   =  (int *)calloc(data->colsSubMat,sizeof(int));
            data->nextInsert[k] =  0;
        }

        data->rowsSubMat=r+1;
        //self->rows = data->rowsSubMat;

    }


    //Controllo se ho spazio nella riga in cui voglio scrivere un nuovo valore.
    if(data->nextInsert[r] == data->colsSubMat){

        data->colsSubMat++;
   //     self->cols = data->colsSubMat;

        //Se ho riempito tutta la riga rialloco le matrici aggiungendo una nuova colonna
        for(int i=0; i < data->rowsSubMat; i++){
            data->matValues[i] =  (double *)realloc(data->matValues[i], sizeof(double)*(data->colsSubMat));
            data->matCols[i]   =  (int *)realloc(data->matCols[i],   sizeof(int)*(data->colsSubMat));
        }

        
    }

    //Salvo il valore modificando opportunamente le due matrici
    data->matValues[r][data->nextInsert[r]] = val ;
    data->matCols[r][data->nextInsert[r]] = c ;

    //Aggiorno indici per le successive scritture
    data->nextInsert[r]++;

    //incremento numeri non zero
    self->numNonZero++;

    // aggiorno il numero di righe e colonne della matrice
    if(r+1 > self->rows)
        self->rows = r+1;
    if(c+1 > self->cols)
        self->cols = c+1;

    return 0;

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
 * Funzione che permette di stampare tutti i non zero della matrice
*/
void printEllpack(Matrix *self){

    for(int i=0; i<self->numNonZero; i++){
        NotZeroElement* nze = self->getNonZero(self,i);
        logMsg(LOG_TAG_I, "Riga:%d , Colonna:%d , Valore:%f\n",nze->row, nze->col, nze->value);
    }
    
}

/**
 * Funzione che restituisce un non zero. L'assunzione è che tutti i valori non zero della matrice siano 
 * memorizzati come se fossero in un vettore unidimensionale. Quindi specificandoo un indice è possibile
 * ottenere un valore.
*/
NotZeroElement* getNonZeroEllpack(Matrix *self, int numZero){

    DataEllpack* data = (DataEllpack*)self->data;
    NotZeroElement* notZeroElement = (NotZeroElement *)calloc(1,sizeof(NotZeroElement));

    int current=0;

    //Scorro tutti gli elementi della matrice ellpack
    for(int i =0; i < data->rowsSubMat;i++){
        for(int j=0; j< data->colsSubMat; j++){

            //Escludo padding
            if(data->matValues[i][j]!=0){
              
                //Quando l'elemento corrente è quello richiesto dal metodo lo restituisco
                if(current == numZero){
                    notZeroElement->row = i;
                    notZeroElement->col = data->matCols[i][j];
                    notZeroElement->value = data->matValues[i][j];
                    return notZeroElement;
                }

                //Incremento elemento corrente
                current++;
            }
            
        }

    }

    return NULL;
}

long getSizeEllpack(Matrix *self){

    DataEllpack *data = (DataEllpack*)self->data;
    return sizeof(double) * data ->colsSubMat * data ->rowsSubMat + sizeof(int) * data ->colsSubMat * data ->rowsSubMat;
}

/**
 * Funzione per deallocare matrice in memoria
*/
void freeMatrixEllpack(Matrix *self){
    free(self->data);
    free(self);
}

Matrix *cloneEmptyEllpack(Matrix *self){
    Matrix* clone = newMatrixEllpack();
    return clone;
}

/**
 * Costruttore della matrice in formato ellpack
*/
Matrix* newMatrixEllpack() {

    Matrix* matrix = newMatrix();
    DataEllpack* dataEllpack = (DataEllpack*)calloc( 1,sizeof(DataEllpack ));

    dataEllpack->colsSubMat=1; // Inizializzo a 1 il numero di colonne della matrice sparsa
    dataEllpack->rowsSubMat=1; // Inizializzo a 1 il numero di righe
    dataEllpack->nextInsert = (int *)calloc(1,sizeof(int)); 
    
    dataEllpack->matValues = (double **)calloc(1,sizeof(double*));
    dataEllpack->matCols =  (int **) calloc(1,sizeof(int*));

    //Inizialmente avrò una matrice con una sola riga e una sola colonna
    dataEllpack->matValues[0] = (double *)calloc(1,sizeof(double));
    dataEllpack->matCols[0] = (int *)calloc(1,sizeof(int));
    

    //Inizializzo matrice da resitituire
    matrix->formatName = "ELLPACK";
    matrix->data = dataEllpack;
    matrix->put = putEllpack;
    matrix->get = getEllpack;
    matrix->print = printEllpack;
    matrix->getNonZero = getNonZeroEllpack;
    matrix->getSize = getSizeEllpack;
    matrix->cloneEmpty = cloneEmptyEllpack;
    matrix->cols = dataEllpack->colsSubMat;
    matrix->rows = dataEllpack->rowsSubMat;
    matrix->numNonZero = 0;

    return matrix;
}

