#ifndef MATRIX_H
#define MATRIX_H

#include "logger/logger.h"
#include <stdbool.h>

/**
 * Struttura che impacchetta un elemento non zero di una matrice sparsa.
*/
typedef struct notZeroElement
{
    int row;
    int col;
    double value;

}NotZeroElement;


/**
 * A generic Matrix object.
 * Implementations must populate members according
 * to their format.
*/
typedef struct matrix
{

    int rows;
    int cols;

    /**
     * Stringa identificativa del formato
    */
    const char *formatName; 
    
    /**
     * Matrix data. The exact data structure is chosen by the
     * format.
    */
    void *data;

    /**
     * num of noeh n-zero elements
     */
    int numNonZero;

    /**
     * returns the value at given coordinates.
     * Return NaN if an error occurred.
     */
    double (*get)(struct matrix *self, int r, int c);

    /**
     * returns the non-zero value at given position,
     * as if the non-zero values were put in a row.
     * Return NULL if an error occurred.
    */
    NotZeroElement* (*getNonZero)(struct matrix *self, int pos);

    /**
     * puts a value at given coordinates.
     * Returns non-zero if an error occurred.
     */
    int (*put)(struct matrix *self, int r, int c, double val);

    /**
     * Prints the matrix to screen
    */
    void (*print)(struct matrix *self);

    /**
     * Returns an approximate size in bytes of the data used by this format to
     * store the matrix values and coordinates.
     * Size is not exact since it can neglect some overhead data.
    */
    long (*getSize)(struct matrix *self);

    /**
     * Creates an empty Matrix object in the same format as self. (See PROTOTYPE pattern).
     * Users can use the MEDIATOR to populate the newborn matrix with values from self.
    */
    struct matrix *(*cloneEmpty)(struct matrix *self);

} Matrix;

/**
 * Creates a new Matrix object with NULL data and default method implementations and attribute values.
 * A so created Matrix object should be used only by implementation constructors, which will populate
 * it with the correct data in the desired format, attribute values and method implementations.
*/
Matrix *newMatrix();

/**
 * Frees the memory allocated by the matrix.
*/
void freeMatrix(Matrix *self);

/**
 * Checks if given indexes are out of matrix bounds.
*/
bool outOfBounds(Matrix *self, int r, int c);

#endif // MATRIX_H