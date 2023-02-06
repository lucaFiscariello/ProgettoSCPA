#include "logger/logger.h"
#include "matrix/matrix.h"
#include "matrix/formats/matrixEllpack.h"
#include "matrix/formats/multiVector.h"
#include "product/product.h"
#include "matrix/formats/coo.h"
#include "matrix/formats/mm/mm.h"
#include "mediator/mediator.h"

void testProduct(int (*productSparseMultivector)(Matrix *m1, Matrix *m2, Matrix *mr, Sample *s) ){
    
    char *matrixFileName = "/data/dlaprova/matrix-multiVector-product/matrixFile/Trec5.mtx";
    
    Matrix *matrixMM = newMatrixMM(matrixFileName);
    Matrix *matrixEllpack = newMatrixEllpack();
    Matrix *multiVector = newMultiVector( matrixMM->cols,5 /*qualsiasi valore*/);

    MatrixSampleID m1sid, m2sid;

    DataEllpack *dataEllpack = (DataEllpack *) matrixEllpack->data;
    
    Sample *samplePar = (Sample *)calloc(1,sizeof(Sample));
    Sample *sampleSer = (Sample *)calloc(1,sizeof(Sample));

    //Riempio multivettore
    for(int i=0; i< multiVector->rows;i++)
        for(int j= 0; j< multiVector->cols; j++)
            multiVector->put(multiVector,i,j,i+j);

    convert(matrixMM,matrixEllpack);
    
    m1sid.name = matrixFileName;
    m1sid.numBytes = matrixEllpack ->getSize(matrixEllpack);
    m1sid.numElements = matrixEllpack ->numNonZero;

    m2sid.name = "multiVector";
    m2sid.numBytes = multiVector ->getSize(multiVector);
    m2sid.numElements = multiVector ->cols;

    samplePar ->m1SampleId = &m1sid;
    samplePar ->m2SampleId = &m2sid;
    sampleSer ->m1SampleId = &m1sid;
    sampleSer ->m2SampleId = &m2sid;
    
    Matrix *resultPar= newMultiVector(matrixEllpack->rows,multiVector->cols);
    Matrix *resultSer= newMultiVector(matrixEllpack->rows,multiVector->cols);

    productSparseMultivector(matrixEllpack,multiVector,resultPar,samplePar);
    calcGflops(samplePar);
    calcBandwidth(samplePar);

    productMatrixMatrixSerial(matrixEllpack,multiVector,resultSer,sampleSer);
    calcGflops(sampleSer);
    calcBandwidth(sampleSer);

    printf( "\n%s\n", samplePar ->productName);
    resultPar->print(resultPar);
    printf( "GigaFlop: %f\n\n", samplePar->gflops);
    printf("bandwidth: %f\n",samplePar->bandwidth);

    printf( "\n%s\n", sampleSer ->productName);
    resultSer->print(resultSer);
    printf( "GigaFlop: %f\n\n", sampleSer->gflops);
    printf("bandwidth: %f\n",sampleSer->bandwidth);

    freeMatrixMM(matrixMM);
    freeMatrixEllpack(matrixEllpack);
}


/*************************************** MAIN ****************************************************/


int main(int argc, char *argv[]){
    
    testProduct(productEllpackMultivectorParallelCPU);
    //testProduct(productMatrixMatrixParallelEllpack);

    return 0;
}