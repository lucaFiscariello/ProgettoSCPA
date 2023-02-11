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

/*************************************** TEST FORMATI MATRICE ****************************************************/

void testMatrixEllpack(){
    
    Matrix* matrix = newMatrixEllpack();

     for(int i=0;i<9;i++)
        for(int j=0;j<9;j++){
            matrix->put(matrix,i,j,i+j);
        }


    logMsg(LOG_TAG_I, "Letto il valore: %f\n",matrix->get(matrix,2,2));

    logMsg(LOG_TAG_I, "Letto il valore: %f\n",matrix->get(matrix,1,1));
    logMsg(LOG_TAG_I, "Letto il valore: %f\n",matrix->get(matrix,1,2));
    logMsg(LOG_TAG_I, "Letto il valore: %f\n",matrix->get(matrix,2,2));
    logMsg(LOG_TAG_I, "Letto il valore: %f\n",matrix->get(matrix,3,3));
    logMsg(LOG_TAG_I, "Letto il valore: %f\n",matrix->get(matrix,3,2));
    logMsg(LOG_TAG_I, "LOG_TAG_I non zero sono: %d\n",matrix->numNonZero);

    logMsg(LOG_TAG_I, "Primo elemento non zero: %f\n",matrix->getNonZero(matrix,0)->value);
    logMsg(LOG_TAG_I, "Secondo elemento non zero: %f\n",matrix->getNonZero(matrix,1)->value);

    matrix->print(matrix);
    freeMatrixEllpack(matrix);
    
}
void testMatrixCOO(){

    Matrix *matrix = newMatrixCOO();
    
    matrix->put(matrix,1,1,1.0);
    matrix->print(matrix);
    logMsg(LOG_TAG_D, "Valore inserito in posizione 1, 1: %f\n", matrix->get(matrix,1,1));
    logMsg(LOG_TAG_D, "Valore in posizione 2, 2: %f\n", matrix->get(matrix,2,2));
    logMsg(LOG_TAG_D, "Non zeri: %d\n", matrix->numNonZero);

    matrix->put(matrix,1,1,0.0);
    matrix->print(matrix);

    matrix->put(matrix,2,2,2.0);
    matrix->put(matrix,3,3,3.0);

    logMsg(LOG_TAG_D, "Primo elemento non zero: %f\n", matrix->getNonZero(matrix,0)->value);
    logMsg(LOG_TAG_D, "Secondo elemento non zero: %f\n", matrix->getNonZero(matrix,1)->value);

    freeMatrixCOO(matrix);
}

void testMultiVector(){

    Matrix *multivector = newMultiVector(4,4);

    multivector->put(multivector,1,1,1.0);
    multivector->put(multivector,1,2,5.0);
    multivector->put(multivector,2,2,2.0);
    multivector->put(multivector,3,3,3.0);

    multivector->print(multivector);
    freeMultiVector(multivector);
}

void testMatrixMM(){

    Matrix *matrix = newMatrixMM("matrixFile/Trec5.mtx");
    matrix->print(matrix);
    logMsg(LOG_TAG_D, "matrix[%d][%d] = %f\n", 1, 1, matrix->get(matrix,1,1));
    NotZeroElement *nze;
    for (int i = 0; i < 5; i ++){
        nze = matrix->getNonZero(matrix,i);
        logMsg(LOG_TAG_D, "nonZero in position %d =  %d %d %f\n", i, nze->row, nze->col, nze->value);
    }
    freeMatrixMM(matrix);
}

void testMMPatternSymmetric(){
    Matrix *m = newMatrixMM("matrixFile/bcspwr01.mtx");
    m->print(m);
    logMsg(LOG_TAG_D, "matrix[%d][%d] = %f\n", 0, 1, m->get(m,1,1));
    NotZeroElement *nze;
    for (int i = 0; i < 5; i ++){
        nze = m->getNonZero(m,i);
        logMsg(LOG_TAG_D, "nonZero in position %d =  %d %d %f\n", i, nze->row, nze->col, nze->value);
    }
    freeMatrixMM(m);
}


void testMatrixMediatorCooToEll(){

    Matrix *matrixCoo = newMatrixCOO();
    Matrix *matrixEll = newMatrixEllpack();

    matrixCoo->put(matrixCoo,1,1,1.0);
    matrixCoo->put(matrixCoo,1,2,5.0);
    matrixCoo->put(matrixCoo,2,2,2.0);
    matrixCoo->put(matrixCoo,3,3,3.0);

    convert(matrixCoo,matrixEll);
    matrixEll->print(matrixEll);

    freeMatrixCOO(matrixCoo);
    freeMatrixEllpack(matrixEll);
}

void testMatrixMediatorElltoCoo(){

    Matrix *matrixCoo = newMatrixCOO();
    Matrix *matrixEll = newMatrixEllpack();

    matrixEll->put(matrixEll,2,2,5.0);

    convert(matrixEll,matrixCoo);
    matrixCoo->print(matrixCoo);

    freeMatrixCOO(matrixCoo);
    freeMatrixEllpack(matrixEll);
}

void testMatrixMediatorMMtoCOO(){

    Matrix *matrixMM = newMatrixMM("matrixFile/Trec5.mtx");
    Matrix *matrixCoo = newMatrixCOO();

    convert(matrixMM,matrixCoo);
    matrixCoo->print(matrixCoo);

    freeMatrixMM(matrixMM);
    freeMatrixCOO(matrixCoo);
}


void testMatrixMediatorMMtoEllpack(){

    Matrix *matrixMM = newMatrixMM("matrixFile/cage4.mtx");
    Matrix *matrix = newMatrixEllpack();

    convert(matrixMM,matrix);
    matrix->print(matrix);

    freeMatrixMM(matrixMM);
    freeMatrixEllpack(matrix);
}


void testProductCoo(){

    Matrix *matrixMM = newMatrixMM("matrixFile/Trec5.mtx");
    Matrix *matrixCoo = newMatrixCOO();
    Matrix *multiVector = newMultiVector(matrixMM->cols,matrixMM->rows );
    Matrix *result= newMultiVector(matrixMM->cols,matrixMM->rows );
    Sample *sample = (Sample*) calloc(1,sizeof(Sample));

    //Riempio multivettore
    for(int i=0; i< matrixMM->rows;i++)
        for(int j= 0; j< matrixMM->cols; j++)
            multiVector->put(multiVector,i,j,i+j);
        

    convert(matrixMM,matrixCoo);
    productMatrixMatrixSerial(matrixCoo,multiVector,result,sample);

    //Stampo risultato
    result->print(result);
    
    freeMatrixMM(matrixMM);
    freeMatrixCOO(matrixCoo);
    freeMatrixEllpack(result);
}

void testProductEllpack(){

    Matrix *matrixMM = newMatrixMM("matrixFile/Trec5.mtx");
    Matrix *matrixEllpack = newMatrixEllpack();
    Matrix *multiVector = newMultiVector( matrixMM->cols,3);
    Sample *sample = (Sample *)calloc(1,sizeof(Sample));

    //Riempio multivettore
    for(int i=0; i< multiVector->cols;i++)
        for(int j= 0; j< multiVector->rows; j++)
            multiVector->put(multiVector,i,j,i+j);

    convert(matrixMM,matrixEllpack);

    Matrix *result= newMultiVector(matrixEllpack->cols,multiVector->rows);
    productMatrixMatrixSerial(matrixEllpack,multiVector,result,sample);

    //Stampo risultato
    multiVector->print(multiVector);
    matrixEllpack->print(matrixEllpack);
    result->print(result);

    //verifico se sample memorizza info        
    logMsg(LOG_TAG_D, "Nome funzione eseguita: %s\n", sample->productName);
    logMsg(LOG_TAG_D, "GigaFlop: %f\n", sample->gflops);

    freeMatrixMM(matrixMM);
    freeMatrixEllpack(matrixEllpack);
    freeMatrixEllpack(result);
}

