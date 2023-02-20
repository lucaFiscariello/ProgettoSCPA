#include "product/product.h"
#include <stdlib.h>
#include <stdio.h>


double calcGflops(Sample *self){

    long numNonZero = self->m1SampleId->numElements;
    long nMVCols = self->m2SampleId->numElements;
    time_t execTimeSecs = self->execTimeSecs;
    long execTimeNsecs = self->execTimeNsecs;


    self ->gflops = (double) ((2 * numNonZero * nMVCols) / (execTimeSecs + execTimeNsecs *1e-9)) * 1e-9;

    return self->gflops;
}

double calcBandwidth(Sample *self){

    double numMBytes = (self->m1SampleId->numBytes + self->m2SampleId->numBytes) / 1e6;
    time_t execTimeSecs = self->execTimeSecs;
    long execTimeNsecs = self->execTimeNsecs;
    
    self ->bandwidth = numMBytes / (execTimeSecs + (execTimeNsecs * 1e-9));
    return self->bandwidth;

}

MatrixSampleID *newMatrixSampleID(long numElements, long numBytes, const char *name, const char *formatName){
    MatrixSampleID *self = calloc(1, sizeof(MatrixSampleID));
    self->numElements = numElements;
    self->numBytes = numBytes;
    self->name = name;
    self->formatName = formatName;
    return self;
}
void freeMatrixSampleID(MatrixSampleID *self){
    free(self);
}
