#include "mediator/mediator.h"
#include <stdlib.h>

void convert(Matrix *from, Matrix *to){

    NotZeroElement* nze = calloc(1,sizeof(NotZeroElement));

    //Scorro gli elementi non zero della matrice "from" e li memorizzo in "to"
    for(int i =0; i<from->numNonZero ; i++){

        nze = from->getNonZero(from,i);
        to->put(to, nze->row, nze->col,nze->value);

    }
}
