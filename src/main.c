#include "logger/logger.h"
#include "matrix/matrix.h"
#include "matrix/formats/matrixEllpack.h"
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

    matrixEll->put(matrixEll,1,1,1.0);
    matrixEll->put(matrixEll,1,2,5.0);
    matrixEll->put(matrixEll,2,2,2.0);
    matrixEll->put(matrixEll,3,3,3.0);

    convert(matrixEll,matrixCoo);
    matrixCoo->print(matrixCoo);

    freeMatrixCOO(matrixCoo);
    freeMatrixEllpack(matrixEll);
}

void testMatrixMM(){

    Matrix *matrix = newMatrixMM("/home/daniele/Scaricati/Trec5/Trec5.mtx");
    NotZeroElement *nze;
    matrix->print(matrix);
    logMsg(LOG_TAG_D, "matrix[%d][%d] = %f\n", 1, 1, matrix->get(matrix,1,1));
    nze = matrix->getNonZero(matrix,0);
    logMsg(LOG_TAG_D, "first non zero element: %d %d %f\n", nze ->row, nze->col, nze->value);
    freeMatrixMM(matrix);
}

int main(int argc, char *argv[]){
    
    //testMatrixEllpack();
    //testMatrixCOO();
    testMatrixMM();

    //testMatrixMediatorCooToEll();
    //testMatrixMediatorElltoCoo();

    return 0;
}