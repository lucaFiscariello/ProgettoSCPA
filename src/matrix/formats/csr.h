#ifndef CSR_H
#define CSR_H

#include "matrix/matrix.h"

/**
 * @brief A matrix in CSR format (Compressed Sparse Row) is similar to the
 * COO format. The difference is that row indexes are not stored, instead a 
 * pointer to the first element of each row is stored. Non-zero elements of
 * the same row are stored in order. 
 * 
 */
typedef struct csr_data{

    /**
     * @brief Indexes to the first column of each row.
     * Each position of this array is mapped to a row.
     * In this way we partiton the columns array according to rows. 
     */
    int *firstColOfRowIndexes;

    /**
     * Num of rows stored in the matrix. (Plus the last value, which is not
     * a row, but just the number of non-zero elements)
    */
    int numCompressedRows;

    /**
     * Actual size of the values and columns arrays.
    */
    int valuesCapacity;

    /**
     * @brief Column indexes of the non-zero elements.
     * An element in a given position of this array is the column index of the
     * corresponding element in the values array.
     */
    int *columns;

    /**
     * @brief Values of the non-zero elements.
     * Elements are stored in the same order they appear in the matrix.
     */
    double *values;
} CSRData;

Matrix *newMatrixCSR();
void freeMatrixCSR(Matrix *matrix);

#endif // CSR_H