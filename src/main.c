#include "logger/logger.h"
#include "matrix/matrix.h"
#include "matrix/formats/matrixEllpack.h"
#include "matrix/formats/multiVector.h"
#include "product/product.h"
#include "matrix/formats/coo.h"
#include "matrix/formats/mm/mm.h"
#include "mediator/mediator.h"          


void testMatrixEllpack(){
    
    Matrix* matrix = newMatrixEllpack();

    matrix->put(matrix,1,1,1.0);
    matrix->put(matrix,1,2,5.0);
    matrix->put(matrix,2,2,2.0);
    matrix->put(matrix,3,3,3.0);


    logMsg(LOG_TAG_I, "Letto il valore: %f\n",matrix->get(matrix,1,1));
    logMsg(LOG_TAG_I, "Letto il valore: %f\n",matrix->get(matrix,1,2));
    logMsg(LOG_TAG_I, "Letto il valore: %f\n",matrix->get(matrix,2,2));
    logMsg(LOG_TAG_I, "Letto il valore: %f\n",matrix->get(matrix,3,3));
    logMsg(LOG_TAG_I, "Letto il valore: %f\n",matrix->get(matrix,3,2));
    logMsg(LOG_TAG_I, "LOG_TAG_I non zero sono: %d\n",matrix->numNonZero);

    logMsg(LOG_TAG_I, "Primo elemento non zero: %f\n",matrix->getNonZero(matrix,0)->value);
    logMsg(LOG_TAG_I, "Secondo elemento non zero: %f\n",matrix->getNonZero(matrix,1)->value);

    matrix->print(matrix);
    freeMatrixEllpack(matrix);
    
}

void testMatrixCOO(){

    Matrix *matrix = newMatrixCOO();

    matrix->put(matrix,1,1,1.0);
    matrix->print(matrix);
    logMsg(LOG_TAG_D, "Valore inserito in posizione 1, 1: %f\n", matrix->get(matrix,1,1));
    logMsg(LOG_TAG_D, "Valore in posizione 2, 2: %f\n", matrix->get(matrix,2,2));
    logMsg(LOG_TAG_D, "Non zeri: %d\n", matrix->numNonZero);

    matrix->put(matrix,1,1,0.0);
    matrix->print(matrix);

    matrix->put(matrix,2,2,2.0);
    matrix->put(matrix,3,3,3.0);

    logMsg(LOG_TAG_D, "Primo elemento non zero: %f\n", matrix->getNonZero(matrix,0)->value);
    logMsg(LOG_TAG_D, "Secondo elemento non zero: %f\n", matrix->getNonZero(matrix,1)->value);

    freeMatrixCOO(matrix);
}

void testMatrixMediatorCooToEll(){

    Matrix *matrixCoo = newMatrixCOO();
    Matrix *matrixEll = newMatrixEllpack();

    matrixCoo->put(matrixCoo,1,1,1.0);
    matrixCoo->put(matrixCoo,1,2,5.0);
    matrixCoo->put(matrixCoo,2,2,2.0);
    matrixCoo->put(matrixCoo,3,3,3.0);

    convert(matrixCoo,matrixEll);
    matrixEll->print(matrixEll);

    freeMatrixCOO(matrixCoo);
    freeMatrixEllpack(matrixEll);
}

void testMatrixMediatorElltoCoo(){

    Matrix *matrixCoo = newMatrixCOO();
    Matrix *matrixEll = newMatrixEllpack();

    matrixEll->put(matrixEll,2,2,5.0);

    convert(matrixEll,matrixCoo);
    matrixCoo->print(matrixCoo);

    freeMatrixCOO(matrixCoo);
    freeMatrixEllpack(matrixEll);
}

void testMatrixMediatorMMtoCOO(){

    Matrix *matrixMM = newMatrixMM("matrixFile/Trec5.mtx");
    Matrix *matrixCoo = newMatrixCOO();

    convert(matrixMM,matrixCoo);
    matrixCoo->print(matrixCoo);

    freeMatrixMM(matrixMM);
    freeMatrixCOO(matrixCoo);
}

void testMatrixMM(){

    Matrix *matrix = newMatrixMM("matrixFile/cage4.mtx");
    matrix->print(matrix);
    logMsg(LOG_TAG_D, "matrix[%d][%d] = %f\n", 1, 1, matrix->get(matrix,1,1));
    NotZeroElement *nze;
    for (int i = 0; i < 5; i ++){
        nze = matrix->getNonZero(matrix,i);
        logMsg(LOG_TAG_D, "nonZero in position %d =  %d %d %f\n", i, nze->row, nze->col, nze->value);
    }
    freeMatrixMM(matrix);
}

void testMMPatternSymmetric(){
    Matrix *m = newMatrixMM("matrixFile/bcspwr01.mtx");
    m->print(m);
    logMsg(LOG_TAG_D, "matrix[%d][%d] = %f\n", 0, 1, m->get(m,1,1));
    NotZeroElement *nze;
    for (int i = 0; i < 5; i ++){
        nze = m->getNonZero(m,i);
        logMsg(LOG_TAG_D, "nonZero in position %d =  %d %d %f\n", i, nze->row, nze->col, nze->value);
    }
    freeMatrixMM(m);
}

void testMultiVector(){

    Matrix *multivector = newMultiVector(4,4);

    multivector->put(multivector,1,1,1.0);
    multivector->put(multivector,1,2,5.0);
    multivector->put(multivector,2,2,2.0);
    multivector->put(multivector,3,3,3.0);

    multivector->print(multivector);
    freeMultiVector(multivector);
}

void testProduct(){

    Matrix *matrixMM = newMatrixMM("matrixFile/Trec5.mtx");
    Matrix *matrixCoo = newMatrixCOO();
    Matrix *multiVector = newMultiVector(matrixMM->rows, matrixMM->cols);
    double** result;

    //Riempio multivettore
    for(int i=0; i< matrixMM->rows;i++)
        for(int j= 0; j< matrixMM->cols; j++)
            multiVector->put(multiVector,i,j,i+j);
        

    convert(matrixMM,matrixCoo);
    result = productMatrixMatrixSerial(matrixCoo,multiVector);

    freeMatrixMM(matrixMM);
    freeMatrixCOO(matrixCoo);
    free(result);
}

void testProductNotMM(){

    Matrix* matrix = newMatrixCOO();

    matrix->put(matrix,1,1,1.0);
    matrix->put(matrix,2,2,1.0);
    matrix->put(matrix,3,3,1.0);
    matrix->put(matrix,4,4,1.0);

    Matrix *multiVector = newMultiVector(matrix->rows, matrix->cols);
    double** result;

    //Riempio multivettore
    for(int i=0; i< matrix->rows;i++)
        for(int j= 0; j< matrix->cols; j++)
            multiVector->put(multiVector,i,j,i+j);

    result = productMatrixMatrixSerial(matrix,multiVector);

    //Stampo risultato
    for(int i=0; i< matrix->rows;i++)
        for(int j= 0; j< matrix->cols; j++)
            logMsg(LOG_TAG_D, "result[%d][%d]= %f\n", i, j, result[i][j]);


    freeMatrixCOO(matrix);
    free(result);
}

int main(int argc, char *argv[]){
    
    //testMatrixEllpack();
    //testMatrixCOO();
    //testMatrixMM();
    //testMultiVector();

    //testMMPatternSymmetric();
    //testMatrixMediatorCooToEll();
    //testMatrixMediatorElltoCoo();
    //testMatrixMediatorMMtoCOO();

    testProductNotMM();
    testProduct(); //Versione con BUG

    return 0;
}