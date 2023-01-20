#ifndef MM_H
#define MM_H

#include "matrix/matrix.h"
#include "matrix/formats/mm/mmio.h"
#include <stdio.h>
#include <stdbool.h>

/**
 * The representation of a MatrixMarket sparse matrix data.
*/
typedef struct mm_data{

    FILE *file; /** file pointer to the matrix file*/
    char *filename; /** name of the underlying MatrixMarket file*/
    MM_typecode typecode; /** matrix charateristics*/
    int numValueLines; /** number of lines containing matrix values*/


} DataMM;


/** 
 * constructor.
 * It also opens the underlying file.
 * The matrix file must already exists and be in MatrixMarket format.
 * */
Matrix *newMatrixMM(char *filename);

/** destructor.
 * It also closes the underlying file
*/
void freeMatrixMM(Matrix *self);

char *mmio_strerror(int mm_err);

#define ON_ERROR_LOG_MMIO_AND_RETURN(isError, retVal, msg, ...)\
    do { \
        int err = isError; \
        if (err){ \
            char *errmsg = mmio_strerror(err); \
            char *log_mmio_buf = calloc(strlen(msg) + strlen(LOG_MSG_SEP) + strlen(errmsg) + 2, sizeof(char)); \
            memcpy(log_mmio_buf, msg, strlen(msg)); \
            log_mmio_buf = strcat(log_mmio_buf, LOG_MSG_SEP); \
            log_mmio_buf = strcat(log_mmio_buf, errmsg); \
            log_mmio_buf = strcat(log_mmio_buf, "\n");\
            LOG_ERROR(log_mmio_buf __VA_OPT__(,) __VA_ARGS__); \
            free(log_mmio_buf); \
        } \
    } while(0);

char *objectToString(MM_typecode matcode);
char *sparseDenseToSpring(MM_typecode matcode);
char *dataTypeToString(MM_typecode matcode);
char *storageSchemeToString(MM_typecode matcode);

#endif // MM_H