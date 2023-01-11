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
     * num of non-zero elements
     */
    int numNonZero;

    /**
     * returns the value at given coordinates
     */
    double (*get)(int r, int c);

    /**
     * returns the non-zero value at given position,
     * if the non-zero values were put in a row
    */
    double (*getNonZero)(int pos);

    /**
     * puts a value at given coordinates
     */
    void (*put)(int r, int c, double val);

} Matrix;