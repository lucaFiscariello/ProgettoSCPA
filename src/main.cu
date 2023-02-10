#include "logger/logger.h"
#include "matrix/matrix.h"
#include "matrix/formats/matrixEllpack.h"
#include "matrix/formats/multiVector.h"
#include "product/product.h"
#include "matrix/formats/coo.h"
#include "matrix/formats/mm/mm.h"
#include "mediator/mediator.h"
#include "experimenter/experimenter.h"

void testProduct(int (*productSparseMultivector)(Matrix *m1, Matrix *m2, Matrix *mr, Sample *s) ){
    
    char *matrixFileName = "matrixFile/Trec5.mtx";
    
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

void testExperimenter(){
    char *matrixFileName = "matrixFile/Trec5.mtx";
    
    const int NUM_M1 = 1;
    const int NUM_M2 = 1;
    const int NUM_PRODUCTS = 3;
    const int NUM_TRIALS = 2;
    const int NUM_EXPERIMENTS = NUM_M1 * NUM_M2 * NUM_PRODUCTS * NUM_TRIALS;
    
    Matrix *matrixMM = newMatrixMM(matrixFileName);
    Matrix *matrixEllpack = newMatrixEllpack();
    Matrix *multiVector = newMultiVector( matrixMM->cols,5 /*qualsiasi valore*/);

    MatrixSampleID *m1sids[NUM_M1], *m2sids[NUM_M2];
    for (int i = 0; i < NUM_M1; i ++){
        m1sids[i] = (MatrixSampleID *)calloc(1, sizeof(MatrixSampleID));
    }
    for (int i = 0; i < NUM_M2; i ++){
        m2sids[i] = (MatrixSampleID *)calloc(1, sizeof(MatrixSampleID));
    }

    Sample *samples[10];

    int (*products[NUM_PRODUCTS])(Matrix *, Matrix *, Matrix *, Sample *);
    products[0] = productMatrixMatrixSerial;
    products[1] = productMatrixMatrixParallelEllpack;
    products[2] = productEllpackMultivectorParallelCPU;
    
    //Riempio multivettore
    for(int i=0; i< multiVector->rows;i++)
        for(int j= 0; j< multiVector->cols; j++)
            multiVector->put(multiVector,i,j,i+j);

    convert(matrixMM,matrixEllpack);

    m1sids[0] -> name = matrixFileName;
    m1sids[0] -> numBytes = matrixEllpack ->getSize(matrixEllpack);
    m1sids[0] -> numElements = matrixEllpack ->numNonZero;

    m2sids[0] -> name = "multiVector";
    m2sids[0] -> numBytes = multiVector ->getSize(multiVector);
    m2sids[0] -> numElements = multiVector ->cols;

    // invocare l'experimenter
    doExperiments(&matrixEllpack, m1sids, NUM_M1, &multiVector, m2sids, NUM_M2, products,
     NUM_PRODUCTS, NUM_TRIALS, samples);

    // print samples
    for (int i = 0; i < NUM_EXPERIMENTS; i ++){
        printf("%-50s, %-12s, %-36s, %-6d, %-6.6f, %-6.6f\n", samples[i]->m1SampleId->name,
         samples[i]->m2SampleId->name, samples[i]->productName, samples[i]->trial,
          samples[i]->gflops, samples[i]->bandwidth);
    }

    printSamplesToCSV(NUM_EXPERIMENTS,samples,"sample.csv");
}


/*************************************** MAIN ****************************************************/


int main(int argc, char *argv[]){
    
    //testProduct(productEllpackMultivectorParallelCPU);
    //testProduct(productMatrixMatrixParallelEllpack);
    testExperimenter();

    

    return 0;
}