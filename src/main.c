#include "logger/logger.h"
#include "matrix/matrix.h"
#include "matrix/formats/matrixEllpack.h"
#include "matrix/formats/coo.h"


void testMatrixEllpack(){
    
    Matrix* matrix = newMatrixEllpack(4,4);

    matrix->put(matrix,1,1,1.0);
    matrix->put(matrix,2,2,2.0);

    logMsg(I, "Letto il valore: %f\n",matrix->get(matrix,1,1));
    logMsg(I, "I non zero sono: %d\n",matrix->numNonZero);

}
void testMatrixCOO(){

    Matrix *matrix = newMatrixCOO();

    matrix->put(matrix,1,1,1.0);
    matrix->print(matrix);
    logMsg(D, "Valore inserito in posizione 1, 1: %f\n", matrix->get(matrix,1,1));
    logMsg(D, "Valore in posizione 2, 2: %f\n", matrix->get(matrix,2,2));
    matrix->put(matrix,1,1,0.0);
    matrix->print(matrix);
    freeMatrixCOO(matrix);
}
int main(int argc, char *argv[]){
    
    //testMatrixEllpack();
    testMatrixCOO();

    return 0;
}