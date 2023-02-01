#include "mediator/mediator.h"
#include "matrix/matrix.h"
#include <stdlib.h>


void convert(Matrix *from, Matrix *to){

    NotZeroElement* nze;

    //Verifico se il metodo che restituisce un non zero è implementato
    if(from->getNonZero != NULL){

        //Scorro gli elementi non zero della matrice "from" e li memorizzo in "to"
        for(int i =0; i<from->numNonZero ; i++){        
                nze = from->getNonZero(from,i);
                to->put(to, nze->row, nze->col,nze->value);
        }
        
    }
}

void convert_dense_too(Matrix *from, Matrix *to){

    for (int r = 0; r < from->rows; r++){
        for (int c = 0; c < from->cols; c++){
            to->put(to, r, c, from->get(from, r, c));
        }
    }
}