#include "matrix/formats/mm/mm.h"
#include "logger/logger.h"
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define COMMENT_STARTER_CHAR '%'
#define UNKWOWN "unknown"

char *objectToString(MM_typecode matcode){
    if (mm_is_matrix(matcode)) 
        return MM_MTX_STR;
    else
        return UNKWOWN;
}

char *sparseDenseToSpring(MM_typecode matcode){
    if (mm_is_sparse(matcode))
        return MM_SPARSE_STR;
    else
    if (mm_is_dense(matcode))
        return MM_DENSE_STR;
    else
        return UNKWOWN;
}

char *dataTypeToString(MM_typecode matcode){
    if (mm_is_real(matcode))
        return MM_REAL_STR;
    else
    if (mm_is_complex(matcode))
        return MM_COMPLEX_STR;
    else
    if (mm_is_pattern(matcode))
        return MM_PATTERN_STR;
    else
    if (mm_is_integer(matcode))
        return MM_INT_STR;
    else
        return UNKWOWN;
}

char *storageSchemeToString(MM_typecode matcode){
    if (mm_is_general(matcode))
        return MM_GENERAL_STR;
    else
    if (mm_is_symmetric(matcode))
        return MM_SYMM_STR;
    else 
    if (mm_is_hermitian(matcode))
        return MM_HERM_STR;
    else 
    if (mm_is_skew(matcode))
        return MM_SKEW_STR;
    else
        return UNKWOWN;
}

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
int skipMetadata(FILE *f){

    int c = 0;
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
int seekdata(FILE *f, int pos){

    ON_ERROR_LOG_AND_RETURN(skipMetadata(f), -1, "Couldn't skip metadata");
    for(; pos > 0; pos --){
        ON_ERROR_LOG_AND_RETURN(skipLine(f), -1, "Couldn't skip line %d from file %s", pos, f);
    }

    return 0;
}

bool isSymmetric(DataMM *data)
{
    return mm_is_symmetric(data ->typecode) || mm_is_skew(data ->typecode);
}

int readLine(MM_typecode typecode, FILE* file, int *r, int *c, double *v){

    if (mm_is_pattern(typecode))
    {
        fscanf(file, "%d %d\n", r, c);
        *v = 1;
    }
    else 
    {
        fscanf(file, "%d %d %lg\n", r, c, v);
    }
    ON_ERROR_LOG_ERRNO_AND_RETURN(ferror(file), - 1, "Couldn't read line from file");

    return 0;
}

void swap(int *a, int *b){
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

NotZeroElement *getNonZeroMM(Matrix *self, int pos){
    
    NotZeroElement *nze;
    DataMM *data = (DataMM *) self ->data;
    int r, c;
    double v;
    int line, curPos;

    /**
     * Questa variabile è necessaria solo in caso questo metodo venga invocato su una matrice
     * simmetrica.
     * 
     * Inizialmente assumiamo che le posizioni pari corrispondono agli elementi non-zero
     * del triangolo inferiore di una matrice simmetrica. In questo modo, per ogni riga
     * possiamo dire che se la posizione richiesta è pari restituiamo proprio l'elemento
     * scritto nella riga del file, altrimenti prendiamo
     * il suo speculare nel triangolo superiore.
     * 
     * Se si incontra un elemento non-zero sulla diagonale, è necessario scambiare i ruoli
     * di pari e dispari per evitare di restituirlo sia per le posizioni pari che le dispari.
     * 
     * 1 --> le posizioni pari corrispondono agli elementi non-zero del triangolo inferiore 
     * di una matrice simmetrica, le dispari al triangolo superiore
     * 0 --> le posizioni dispari corrispondono agli elementi non-zero del triangolo inferiore
     * di una matrice simmetrica, le pari al triangolo superiore
    */
    int upperTrianglePos = 1;
    
    // sanity check on pos value
    ON_ERROR_LOG_AND_RETURN((pos < 0 || pos >= self ->numNonZero), NULL, "Out of bounds position %d", pos);

    // Discards all lines until we reach the first data line
    reset(data ->file);
    ON_ERROR_LOG_AND_RETURN(seekdata(data->file, 0), NULL, "Couldn't seek data at pos %d\n", pos);

    // leggiamo le righe fino a quando non raggiungiamo la posizione richiesta
    for (line = 0, curPos = 0; line < data ->numValueLines && curPos <= pos; line ++, curPos ++){
        ON_ERROR_LOG_ERRNO_AND_RETURN(readLine(data->typecode,data->file, &r, &c, &v), NULL, "Couldn't read line %d from file %s", pos, data ->filename);
        
        // special care is needed if the matrix is symmetric
        if (isSymmetric(data)){
            // if we are on the diagonal, we need to swap the roles of even and odd positions
            if (r != c){
                curPos ++;
            } else {
                upperTrianglePos = (upperTrianglePos + 1) % 2;
            }
        }
    }
    
    nze = calloc(1, sizeof(NotZeroElement));
    nze ->row = r - 1;
    nze ->col = c - 1;
    nze ->value = v;

    /**
     * Ritorniamo per le posizioni pari gli elementi nella parte triangolare inferiore,
     * mentre per le dispari gli elementi nella parte triangolare superiore.
     * Si da per scontato che gli elementi salvati nel file siano quelli della parte
     * triangolare inferiore.
    */
    if (isSymmetric(data) && pos % 2 == upperTrianglePos){
        swap(&nze ->row, &nze ->col);
    }

    return nze;
}

double getMM(Matrix *self, int r, int c){

    DataMM *data = (DataMM *) self ->data;
    int curR = -1, curC = - 1;
    double curVal = NAN;

    /**
     * If the matrix is symmetric and the indices are above the diagonal, we swap them.
    */
    if (isSymmetric(data) && c > r){
        swap(&r, &c);
    }

    // MatrixMarket indices start from 1
    r ++;
    c ++;

    /**
     * Ogni riga del file corrisponde a un elemento non-nullo della matrice.
     * Posso scorrere tutte le righe cercando quella che contiene gli indici
     * passati dal chiamante.
     */    
    reset(data ->file);
    ON_ERROR_LOG_AND_RETURN(seekdata(data->file, 0), NAN, "Couldn't seek data\n");
    for (int line = 0; line < data ->numValueLines; line ++){
        
        // Se è pattern devo leggere solo riga e colonna, altrimenti devo anche leggere il valore non-zero
        ON_ERROR_LOG_ERRNO_AND_RETURN(readLine(data->typecode,data->file, &curR, &curC, &curVal), NAN, "Couldn't read line %d from file %s", line, data ->filename);

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
    printf("Symmetric matrices display the lower triangular part only\n");
    printf("%-6s%-6s%-6s\n", "row", "col", "val");
    
    reset(data ->file);
    ON_ERROR_LOG_AND_RETURN(seekdata(data->file, 0), , "Couldn't skip metadata\n");
    for (int line = 0; line < data ->numValueLines; line ++){
        ON_ERROR_LOG_ERRNO_AND_RETURN(readLine(data->typecode,data->file, &r, &c, &v), , "Couldn't read line %d from file %s", line, data ->filename);
        printf("%-6d%-6d%-6f\n", r - 1, c - 1, v);
    }
}

void freeMatrixMM(Matrix *self){
    
    DataMM *data = self ->data;
    if (data -> file != NULL){
        fclose(data ->file);
    }
    free(data);
    freeMatrix(self);
}

Matrix *cleanup(Matrix *self){

    freeMatrixMM(self);
    return NULL;
}

Matrix *newMatrixMM(const char *filename){

    Matrix *self = newMatrix();
    DataMM *data = calloc(1, sizeof(DataMM));

    self ->data = data;
    self ->formatName = "MatrixMarket";
    
    // open file and parses banner
    data->file = fopen(filename, "a+");
    ON_ERROR_LOG_ERRNO_AND_RETURN(data->file == NULL, cleanup(self), "Error opening file %s", filename);
    data ->filename = filename;

    // TODO: if the file is newly created, we must write some basic metadata

    // gets metadata from file
    ON_ERROR_LOG_MMIO_AND_RETURN(mm_read_banner(data ->file, &(data ->typecode)), cleanup(self), "Unable to read banner from file %s", filename);

    // we handle only sparse matrices with Real values for now
    ON_ERROR_LOG_AND_RETURN(!mm_is_matrix(data ->typecode), cleanup(self), "File %s is not a matrix\n", filename);
    ON_ERROR_LOG_AND_RETURN(!mm_is_sparse(data ->typecode), cleanup(self), "Matrix in file %s is not sparse\n", filename);
    ON_ERROR_LOG_AND_RETURN(!mm_is_real(data ->typecode) && !mm_is_integer(data ->typecode) && !mm_is_pattern(data ->typecode), cleanup(self), "Unrecognized data type %s in matrix %s\n", dataTypeToString(data ->typecode), filename);

    // we must esclude hermitians too
    ON_ERROR_LOG_AND_RETURN(!mm_is_general(data ->typecode) && !mm_is_symmetric(data ->typecode) && !mm_is_skew(data ->typecode), cleanup(self), "unsupported storage scheme %s from %s\n",storageSchemeToString(data ->typecode), filename);

    // read matrix sizes
    ON_ERROR_LOG_MMIO_AND_RETURN(mm_read_mtx_crd_size(data ->file, &(self ->rows), &(self ->cols), &(data ->numValueLines)), cleanup(self), "Unable to read matrix sizes from file %s", filename);

    // calculates number of non-zero elements according to the matrix symmetry, if any
    if (mm_is_skew(data ->typecode)){
        self ->numNonZero = data ->numValueLines * 2;
    } else if (mm_is_symmetric(data ->typecode)){
        self ->numNonZero = data ->numValueLines * 2 - self ->rows;
    } else {
        self ->numNonZero = data ->numValueLines;
    }
    
    // setup methods
    self->get = getMM;
    self->getNonZero = getNonZeroMM;
    self->put = putMM;
    self->print = printMM;

    return self;
}