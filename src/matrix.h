#ifndef MATRIX_H
#define MATRIX_H

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
    double (*get)(Matrix *self, int r, int c);

    /**
     * returns the non-zero value at given position,
     * as if the non-zero values were put in a row
    */
    double (*getNonZero)(Matrix *self, int pos);

    /**
     * puts a value at given coordinates
     */
    void (*put)(Matrix *self, int r, int c, double val);

} Matrix;

#endif // MATRIX_H