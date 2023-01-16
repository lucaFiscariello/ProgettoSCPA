#include "matrix/formats/mm/mm.h"
#include "logger/logger.h"
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define COMMENT_STARTER_CHAR '%'

char *mmio_strerror(int mm_err){
    switch(mm_err){
        case MM_COULD_NOT_READ_FILE:
            return "MM_COULD_NOT_READ_FILE";
        case MM_PREMATURE_EOF:
            return "MM_PREMATURE_EOF";
        case MM_NOT_MTX:
            return "MM_NOT_MTX";
        case MM_NO_HEADER:
            return "MM_NO_HEADER";
        case MM_UNSUPPORTED_TYPE:
            return "MM_UNSUPPORTED_TYPE";
        case MM_LINE_TOO_LONG:
            return "MM_LINE_TOO_LONG";
        case MM_COULD_NOT_WRITE_FILE:
            return "MM_COULD_NOT_WRITE_FILE";
        default: 
            ON_ERROR_LOG_AND_RETURN(1, NULL, "Not a mmio error code: %d\n", mm_err);
    }
}

/**
 * Reads a line from file f without storing the content.
*/
int skipLine(FILE *f){
    fscanf(f, "%*[^\n]\n");
    ON_ERROR_LOG_ERRNO_AND_RETURN(ferror(f), -1, "Couldn't skip line");
    return 0;
}

void reset(FILE *f){
    clearerr(f);
    rewind(f);
}

/**
 * Reads MarketMatrix file contained in data until it finds the first data line
*/
int skipMetadata(DataMM *data){

    int c;
    FILE* f = data ->file;

    // skip all comment lines
    do {
        // reads first char of line
        fscanf(f, "%c", (char *)&c);
        ON_ERROR_LOG_ERRNO_AND_RETURN(ferror(f), -1, "Couldn't read first char of line");
        if (c == COMMENT_STARTER_CHAR){
            // Discards the rest of the line
            ON_ERROR_LOG_AND_RETURN(skipLine(f), -1, "Couldn't discard rest of line\n");
        }
    } while (c == COMMENT_STARTER_CHAR);

    // skip line of sizes
    ON_ERROR_LOG_AND_RETURN(skipLine(f), -1, "Couldn't discard size line\n");
    return 0;  
}

/** Sets the stream contained in data to the desired line of the data section. 
 * caller must ensure that pos is in bounds.*/
int seekdata(DataMM *data, int pos){
    
    ON_ERROR_LOG_AND_RETURN(skipMetadata(data), -1, "Couldn't skip metadata");
    for(; pos > 0; pos --){
        ON_ERROR_LOG_AND_RETURN(skipLine(data ->file), -1, "Couldn't skip line %d from file %s", pos, data ->filename);
    }

    return 0;
}


NotZeroElement *getNonZeroMM(Matrix *self, int pos){
    
    NotZeroElement *nze;
    DataMM *data = (DataMM *) self ->data;
    int r, c;
    double v;
    
    // sanity check on pos value
    ON_ERROR_LOG_AND_RETURN((pos < 0 || pos >= self ->numNonZero), NULL, "Out of bounds position %d", pos);
    
    // Discards all lines until we reach the desired position
    reset(data ->file);
    seekdata(data, pos);

    // Now we can read the line
    fscanf(data ->file, "%d %d %lg\n", &r, &c, &v);
    ON_ERROR_LOG_ERRNO_AND_RETURN(ferror(data ->file), NULL, "Couldn't read line %d from file %s", pos, data ->filename);
    
    nze = calloc(1, sizeof(NotZeroElement));
    nze ->row = r - 1;
    nze ->col = c - 1;
    nze ->value = v;
    return nze;
}

double getMM(Matrix *self, int r, int c){

    DataMM *data = (DataMM *) self ->data;

    // MatrixMarket indices start from 1
    r ++;
    c ++;

    /**
     * Ogni riga del file corrisponde a un elemento non-nullo della matrice.
     * Posso scorrere tutte le righe cercando quella che contiene gli indici
     * passati dal chiamante.
     */
    int curR = -1, curC = - 1;
    double curVal = 0;
    reset(data ->file);
    ON_ERROR_LOG_AND_RETURN(seekdata(data, 0), NAN, "Couldn't seek data\n");
    for (int line = 0; line < self ->numNonZero; line ++){
        fscanf(data ->file, "%d %d %lg\n", &curR, &curC, &curVal);
        ON_ERROR_LOG_ERRNO_AND_RETURN(ferror(data ->file), NAN, "Couldn't read line %d from file %s", line, data ->filename);
        if (curR == r && curC == c){
            return curVal;  // beccato! ;)
        }
    }

    return 0;   
}

int putMM(Matrix *self, int r, int c, double val){

    // TODO: implement this method
    LOG_UNIMPLEMENTED_CALL();
    return 0;
}

void printMM(Matrix *self){

    DataMM *data = (DataMM *) self ->data;
    int r, c;
    double v;

    printf("MatrixMarket matrix at %s\n", data ->filename);
    printf("Type: %s\n", mm_typecode_to_str(data ->typecode));
    printf("%d x %d, %d non-zero elements\n", self ->rows, self ->cols, self ->numNonZero);
    printf("Printed indices are adjusted to start from 0\n");
    printf("%-6s%-6s%-6s\n", "row", "col", "val");
    
    reset(data ->file);
    ON_ERROR_LOG_AND_RETURN(seekdata(data, 0), , "Couldn't skip metadata\n");
    for (int line = 0; line < self ->numNonZero; line ++){
        fscanf(data ->file, "%d %d %lg\n", &r, &c, &v);
        ON_ERROR_LOG_ERRNO_AND_RETURN(ferror(data ->file), , "Couldn't read line %d from file %s", line, data ->filename);
        printf("%-6d%-6d%-6f\n", r - 1, c - 1, v);
    }
}

Matrix *newMatrixMM(char *filename){

    Matrix *self = newMatrix();
    DataMM *data = calloc(1, sizeof(DataMM));

    self ->data = data;
    
    // open file and parses banner
    data->file = fopen(filename, "a+");
    ON_ERROR_LOG_ERRNO_AND_RETURN(data->file == NULL, NULL, "Error opening file %s", filename);
    data ->filename = filename;

    // TODO: if the file is newly created, we must write some basic metadata

    // gets metadata from file
    ON_ERROR_LOG_MMIO_AND_RETURN(mm_read_banner(data ->file, &(data ->typecode)), NULL, "Unable to read banner from file %s", filename);

    // we handle only sparse matrices with Real values for now
    ON_ERROR_LOG_AND_RETURN(!mm_is_matrix(data ->typecode), NULL, "File %s is not a matrix\n", filename);
    ON_ERROR_LOG_AND_RETURN(!mm_is_sparse(data ->typecode), NULL, "Matrix in file %s is not sparse\n", filename);
    ON_ERROR_LOG_AND_RETURN(!mm_is_real(data ->typecode) && !mm_is_integer(data ->typecode), NULL, "Matrix in file %s has non-real values\n", filename);

    /**
     * TODO: for now, we handle only general matrices.
     * We must add support for symmetric and pattern matrices
    */
    ON_ERROR_LOG_AND_RETURN(!mm_is_general(data ->typecode), NULL, "Matrix in file %s is not general\n", filename);

    // read matrix sizes
    ON_ERROR_LOG_MMIO_AND_RETURN(mm_read_mtx_crd_size(data ->file, &(self ->rows), &(self ->cols), &(self ->numNonZero)), NULL, "Unable to read matrix size from file %s", filename);
    
    // setup methods
    self->get = getMM;
    self->getNonZero = getNonZeroMM;
    self->put = putMM;
    self->print = printMM;
    return self;
}

void freeMatrixMM(Matrix *self){
    
    DataMM *data = self ->data;
    fclose(data ->file);
    free(data);
    freeMatrix(self);
}
