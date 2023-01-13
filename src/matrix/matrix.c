#include "matrix/matrix.h"
#include <stdlib.h>

Matrix *newMatrix(){
    return calloc(1, sizeof(Matrix));
}

void freeMatrix(Matrix *self){
    free(self);
}