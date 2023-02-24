#include "mediator/mediator.h"
#include "matrix/matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include "logger/logger.h"
#include <math.h>



#define COMMENT_STARTER_CHAR '%'


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


void convertFromMM(DataMM *dataMM, Matrix *matrixTo){

    FILE* fileFrom = dataMM ->file;
    int numValueLines = dataMM ->numValueLines;
    char* filename = dataMM ->filename;
    int curR = -1, curC = - 1;
    double curVal = NAN;
    int progress, oldProgress;

    reset(fileFrom);
    ON_ERROR_LOG_AND_RETURN(seekdata(fileFrom, 0), 0, "Couldn't seek data\n");

    //Ogni riga del file corrisponde a un elemento non-nullo della matrice.  
    for (int line = 0; line <numValueLines; line ++){
        
        // Se è pattern devo leggere solo riga e colonna, altrimenti devo anche leggere il valore non-zero
        ON_ERROR_LOG_ERRNO_AND_RETURN(readLine(dataMM ->typecode, fileFrom, &curR, &curC, &curVal), NAN, "Couldn't read line %d from file %s", line, filename);
        matrixTo->put(matrixTo,--curR,--curC,curVal);

        // Se la matrice è simmetrica o skew memorizzo nel formato anche i valori simmetrici
        if (mm_is_skew(dataMM ->typecode) && curC != curR){
            matrixTo->put(matrixTo,curC,curR,-curVal);
        }

        else if(mm_is_symmetric(dataMM ->typecode)&& curC != curR){
            matrixTo->put(matrixTo,curC,curR,curVal);
        }

        progress = (line * 100 / numValueLines);
        if (progress != oldProgress)
        {
            oldProgress = progress;
            logMsg(LOG_TAG_D, "Read %d percent of nzes from file\n", progress);
        }


    }

    return 0;   

}

void convert_dense_too(Matrix *from, Matrix *to){

    int total = from->rows * from->cols;
    int current = 0;
    
    for (int r = 0; r < from->rows; r++){
        for (int c = 0; c < from->cols; c++){
            current = r * from->cols + c + 1;
            to->put(to, r, c, from->get(from, r, c));
            if (current % 10000 == 0 || current == total)
            logMsg(LOG_TAG_D, "Converting %d/%d\n", current, total);
        }
    }
}