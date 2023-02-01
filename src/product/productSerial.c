#include "product/product.h"
#include <string.h>
#include <time.h>



int productMatrixMatrixSerial(Matrix* matrix1, Matrix* matrix2,Matrix *mResult, Sample *sample){

    NotZeroElement* nze;
    int finalRows = matrix1->rows;
    int finalCols = matrix2->cols;

    //Strutture usate nella misurazione delle prestazioni
    struct timespec  tStart;
    struct timespec  tEnd;

    //Inizializzo matrice risultato
    double** result = calloc(finalRows, sizeof(double*));
    for(int i =0; i<finalRows;i++){
        result[i] = calloc(finalCols, sizeof(double));
    }


    clock_gettime(CLOCK_REALTIME,&tStart);

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
            result[nze->row][j] += nze->value * matrix2->get(matrix2,nze->col,j);
        }
    }
    
    clock_gettime(CLOCK_REALTIME ,&tEnd);      


    //Salvo matrice risultato in un formato di matrice dato in input
    for(int i=0;i<finalRows;i++)
        for(int j=0;j<finalCols;j++){
            mResult->put(mResult,i,j,result[i][j]);
        }

    
    //Popolo sample
    sample->execTimeSecs = tEnd.tv_sec - tStart.tv_sec;
    sample->execTimeNsecs = tEnd.tv_nsec - tStart.tv_nsec;
    sample->gflops= calcGflops(sample -> execTimeSecs, sample -> execTimeNsecs, matrix1->numNonZero, matrix2->cols);
    // bandwith computation is approximated since we don't know the size in bytes of the
    // actual formats.
    double size = ((sizeof(double) * matrix1->numNonZero) + (sizeof(double) * matrix2 ->rows * matrix2 ->cols)) / 1e6;
    sample->bandwidth = calcBandwidth(size, sample -> execTimeSecs, sample -> execTimeNsecs);
    sample->productName = (char *) __func__;
    
    return 0;
}
