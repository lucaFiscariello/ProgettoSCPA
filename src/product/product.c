#include "product/product.h"
#include <stdlib.h>

double calcGflops(Sample *self){

    long numNonZero = self->m1SampleId->numElements;
    long nMVCols = self->m2SampleId->numElements;
    time_t execTimeSecs = self->execTimeSecs;
    long execTimeNsecs = self->execTimeNsecs;

    self ->gflops = ((2 * numNonZero * nMVCols) / (execTimeSecs + execTimeNsecs *1e-9)) * 1e-9;
    return self->gflops;
}

double calcBandwidth(Sample *self){

    double numMBytes = (self->m1SampleId->numBytes + self->m2SampleId->numBytes) / 1e6;
    time_t execTimeSecs = self->execTimeSecs;
    long execTimeNsecs = self->execTimeNsecs;
    
    self ->bandwidth = numMBytes / (execTimeSecs + (execTimeNsecs * 1e-9));
    return self->bandwidth;

}