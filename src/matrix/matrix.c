#include "matrix/matrix.h"
#include <stdlib.h>

Matrix *newMatrix(){
    return calloc(1, sizeof(Matrix));
}

void freeMatrix(Matrix *self){
    free(self);
}

bool outOfBounds(Matrix *self, int r, int c){
    return (r < 0 || r >= self ->rows || c < 0 || c >= self ->cols);
}