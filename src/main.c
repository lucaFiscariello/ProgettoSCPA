#include "logger/logger.h"
#include "matrix/matrix.h"
#include "matrix/formats/matrixEllpack.h"
#include "matrix/formats/coo.h"
#include "mediator/mediator.h"


void testMatrixEllpack(){
    
    Matrix* matrix = newMatrixEllpack();

    matrix->put(matrix,1,1,1.0);
    matrix->put(matrix,1,2,5.0);
    matrix->put(matrix,2,2,2.0);
    matrix->put(matrix,3,3,3.0);


    logMsg(I, "Letto il valore: %f\n",matrix->get(matrix,1,1));
    logMsg(I, "Letto il valore: %f\n",matrix->get(matrix,1,2));
    logMsg(I, "Letto il valore: %f\n",matrix->get(matrix,2,2));
    logMsg(I, "Letto il valore: %f\n",matrix->get(matrix,3,3));
    logMsg(I, "Letto il valore: %f\n",matrix->get(matrix,3,2));
    logMsg(I, "I non zero sono: %d\n",matrix->numNonZero);

    logMsg(I, "Primo elemento non zero: %f\n",matrix->getNonZero(matrix,0)->value);
    logMsg(I, "Secondo elemento non zero: %f\n",matrix->getNonZero(matrix,1)->value);

    matrix->print(matrix);
    freeMatrixEllpack(matrix);
    
}

void testMatrixCOO(){

    Matrix *matrix = newMatrixCOO();

    matrix->put(matrix,1,1,1.0);
    matrix->print(matrix);
    logMsg(D, "Valore inserito in posizione 1, 1: %f\n", matrix->get(matrix,1,1));
    logMsg(D, "Valore in posizione 2, 2: %f\n", matrix->get(matrix,2,2));
    logMsg(D, "Non zeri: %d\n", matrix->numNonZero);

    matrix->put(matrix,1,1,0.0);
    matrix->print(matrix);

    matrix->put(matrix,2,2,2.0);
    matrix->put(matrix,3,3,3.0);

    logMsg(D, "Primo elemento non zero: %f\n", matrix->getNonZero(matrix,0)->value);
    logMsg(D, "Secondo elemento non zero: %f\n", matrix->getNonZero(matrix,1)->value);

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

int main(int argc, char *argv[]){
    
    //testMatrixEllpack();
    //testMatrixCOO();

    //testMatrixMediatorCooToEll();
    testMatrixMediatorElltoCoo();


    return 0;
}