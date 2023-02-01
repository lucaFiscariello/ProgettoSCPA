#include "product/product.h"

double calcGflops(time_t execTimeSecs, long execTimeNsecs, int numNonZero, int nMVCols){

    return ((2 * numNonZero * nMVCols) / (execTimeSecs + execTimeNsecs *1e-9)) * 1e-9;
}

double calcBandwidth(double numMBytes, time_t execTimeSecs, long execTimeNsecs){

    return numMBytes / (execTimeSecs + (execTimeNsecs * 1e-9));

}