#include "logger/logger.h"
#include "matrix/matrix.h"
#include "matrix/formats/matrixEllpack.h"
#include "matrix/formats/multiVector.h"
#include "product/product.h"
#include "matrix/formats/coo.h"
#include "matrix/formats/mm/mm.h"
#include "mediator/mediator.h"


void testProductParallelEllpack(){

    Matrix *matrixMM = newMatrixMM("matrixFile/Trec5.mtx");
    Matrix *matrixEllpack = newMatrixEllpack();
    Matrix *multiVector = newMultiVector( matrixMM->cols,5 /*qualsiasi valore*/);
    
    Sample *samplePar = (Sample *)calloc(1,sizeof(Sample));
    Sample *sampleSer = (Sample *)calloc(1,sizeof(Sample));

    //Riempio multivettore
    for(int i=0; i< multiVector->rows;i++)
        for(int j= 0; j< multiVector->cols; j++)
            multiVector->put(multiVector,i,j,i+j);

    convert(matrixMM,matrixEllpack);
    
    Matrix *resultPar= newMultiVector(matrixEllpack->rows,multiVector->cols);
    Matrix *resultSer= newMultiVector(matrixEllpack->rows,multiVector->cols);

    productMatrixMatrixParallelEllpack(matrixEllpack,multiVector,resultPar,samplePar);
    productMatrixMatrixSerial(matrixEllpack,multiVector,resultSer,sampleSer);

    printf( "\n\nprodotto seriale\n");
    resultPar->print(resultPar);
    printf( "GigaFlop: %f\n\n", samplePar->gflops);

    printf( "\n\nprodotto parallelo\n");
    resultSer->print(resultSer);
    printf( "GigaFlop: %f\n\n", sampleSer->gflops);


    freeMatrixMM(matrixMM);
    freeMatrixEllpack(matrixEllpack);

}




/*************************************** MAIN ****************************************************/


int main(int argc, char *argv[]){
    
    testProductParallelEllpack();

    return 0;
}