#ifndef MATRIX_H
#define MATRIX_H


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
     * Matrix data. The exact data structure is chosen by the
     * format.
    */
    void *data;

    /**
     * num of non-zero elements
     */
    int numNonZero;

    /**
     * returns the value at given coordinates
     */
    double (*get)(struct matrix *self, int r, int c);

    /**
     * returns the non-zero value at given position,
     * as if the non-zero values were put in a row
    */
    NotZeroElement* (*getNonZero)(struct matrix *self, int pos);

    /**
     * puts a value at given coordinates
     */
    void (*put)(struct matrix *self, int r, int c, double val);

    /**
     * Prints the matrix to screen
    */
    void (*print)(struct matrix *self);

} Matrix;


/**
 * Creates a new Matrix object.
 * The format is chosen by the implementation.
*/
Matrix *newMatrix();


/**
 * Frees the memory allocated by the matrix.
*/
void freeMatrix(Matrix *self);

#endif // MATRIX_H