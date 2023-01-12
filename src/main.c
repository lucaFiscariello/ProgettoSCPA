#include "header/logger.h"
#include "header/matrixEllpack.h"

int main(int argc, char *argv[]){

    Matrix* matrix = newMatrixEllpack(4,4);

    matrix->put(matrix,1,1,1.0);
    matrix->put(matrix,2,2,2.0);

    logMsg(I, "Letto il valore: %f\n",matrix->get(matrix,1,1));
    logMsg(I, "I non zero sono: %d\n",matrix->numNonZero);

    return 0;
}