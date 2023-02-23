#include "logger/logger.h"
#include "matrix/matrix.h"
#include "matrix/formats/matrixEllpack.h"
#include "matrix/formats/csr.h"
#include "matrix/formats/multiVector.h"
#include "matrix/formats/mm/mm.h"
#include "product/product.h"
#include "matrix/formats/coo.h"
#include "matrix/formats/mm/mm.h"
#include "mediator/mediator.h"
#include "random/rvgs.h"
#include "random/rngs.h"
#include <stdlib.h>
#include <string.h>

/*************************************** PARAMETERS ****************************************************/

// TODO: read following parameters from files

/**
 * Matrix file names to use in the experiments as sparse matrices
 */
const char *MATRIX_FILE_NAMES[] = {
    
    "matrixFile/bcspwr01.mtx",
    "matrixFile/Trec5.mtx",
    "matrixFile/cage4.mtx",
    /*
    "matrixFile/bcspwr01.mtx",
    "matrixFile/west2021.mtx",
    "matrixFile/olm1000.mtx",
    "matrixFile/thermal1.mtx",
    "matrixFile/mac_econ_fwd500.mtx",
    "matrixFile/cant.mtx",
    "matrixFile/nlpkkt80.mtx",
    "matrixFile/adder_dcop_32.mtx",
    "matrixFile/af_1_k101.mtx",
    "matrixFile/af23560.mtx",
    "matrixFile/amazon0302.mtx",
    "matrixFile/bcsstk17.mtx",
    "matrixFile/cavity10.mtx",
    "matrixFile/dc1.mtx",
    "matrixFile/FEM_3D_thermal1.mtx",
    "matrixFile/lung2.mtx",
    "matrixFile/mcfe.mtx",
    "matrixFile/mhd4800a.mtx",
    "matrixFile/olafu.mtx",
    "matrixFile/PR02R.mtx",
    "matrixFile/raefsky2.mtx",
    "matrixFile/rdist2.mtx",
    "matrixFile/roadNet-PA.mtx",
    "matrixFile/thermal2.mtx",
    "matrixFile/thermomech_TK.mtx",
    "matrixFile/webbase-1M.mtx"
    */
    // more matrix file names here ...
};
const int NUM_MATRIX_FILE_NAMES = sizeof(MATRIX_FILE_NAMES) / sizeof(void *); // sizeof su array sullo stack restituisce la memoria occupata dall'array in bytes. (NON FUNZIONA SU POINTERS!). Inoltre uso sizeof(void *) perché i puntatori sono tutti grandi uguale.

/**
 * Sparse matrix formats to use in experiments
 * FIXME: c++ magic is needed to initialize matrix objects in constants, so this
 * code can only compile with nvcc or g++ (gcc won't work).
 * FIXME: dato che ogni prodotto è pensato per lavorare con un solo formato,
 * bisogna fare in modo che ogni prodotto riceva il formato giusto.
 * Dato che per ora non abbiamo un modo per farlo, per ora lasciamo solo ELLPACK.
*/
const Matrix *MATRIX_FORMATS[] = {
    //newMatrixEllpack(),
    newMatrixCSR()
};
const int NUM_MATRIX_FORMATS = sizeof(MATRIX_FORMATS) / sizeof(void *);

/**
 *  Width in columns (k) of the multivector
*/
const int MV_WIDTHS[] = {
    3,
    4,
    8,
    12,
    16,
    32,
    64,
    // more multivector widths here ...
};
const int NUM_MV_WIDTHS = sizeof(MV_WIDTHS) / sizeof(int);

/**
 * All product functions to test
 * TODO: read product names from file and map them to the corresponding function
*/
int (*PRODUCTS[])(Matrix *, Matrix *, Matrix *, Sample *) = {
    //productMatrixMatrixSerial,
    //productMatrixMatrixParallelEllpack,
    //productEllpackMultivectorParallelCPU,
    //productCsrMultivectorParallelCPU
    productCsrMultivectorParallelGPU
    // more product functions here ...
};
const int NUM_PRODUCTS = sizeof(PRODUCTS) / sizeof(void *);

// TODO: read following parameters from command line

/**
 * Number of trials to run for each experiment
*/
const int TRIALS = 20;

/**
 * Seed da usare per la generazione dei numeri casuali
 * Guardare nella funzione PutSeed() per la semantica dei valori.
*/
const long SEED = 123456789;

const double MV_MAX_VAL = 1;
const double MV_MIN_VAL = 0;

/*************************************** HELPERS ****************************************************/

/**
 * Genera un multivettore con un numero di righe pari al numero delle colonne
 * della matrice in ingresso, e un numero di colonne pari a k.
 * Ogni valore del multivettore è generato casualmente con una distribuzione uniforme.
*/
Matrix *craftUniformMultiVectorFromMatrix(Matrix *m, int k, double minVal, double maxVal){

    Matrix *mv = newMultiVector(m -> cols, k);
    for (int r = 0; r < mv -> rows; r++){
        for (int c = 0; c < mv -> cols; c++){
            mv ->put(mv, r, c, Uniform(minVal, maxVal));
        }
    }

    return mv;
}

/**
    Esegue il prodotto matriciale prendendo a coppie le matrici da m1 e m2 in ordine, usando
    tutte le funzioni di prodotto passate in input, e
 * registra le informazioni di prestazioni in un oggetto Sample, uno per ogni prova
 * eseguita, nell'array samples.
 * @param m1 array di matrici 1
 * @param msid1 array di informazioni utili al campionamento che identificano le matrici m1.
 * Devono essere nello stesso ordine e numero di m1.
 * @param m2 array di matrici 2
 * @param msid2 array di informazioni utili al campionamento che identificano le matrici m2.
 * Devono essere nello stesso ordine e numero di m2.
 * @param numM numero di matrici m1 ed m2.
 * @param products array di puntatori alle funzioni che implementano il prodotto matriciale.
 * @param numProducts numero di prodotti matriciali
 * @param numTrials numero di volte per cui ogni esperimento deve essere ripetuto.
 * @param samples array multidimensionale in cui verranno scritti i puntatori ai Samples prodotti
 * da ogni
 * esperimento. Deve avere dimensione almeno pari a numM * numProducts * numTrials.
 * I samples sono allocati dinamicamente, e l'ownership è trasferita al caller.
*/
int doExperiments(
    Matrix *m1[], MatrixSampleID *msid1[],
    Matrix *m2[], MatrixSampleID *msid2[],
    int numM,
    int (*products[])(Matrix *, Matrix *, Matrix *, Sample *), int numProducts,
    int numTrials,
    Sample *samples[])
{
    
    // usiamo un buffer di matrici COO per memorizzare temporaneamente
    // i risultati dei prodotti matriciali, tanto non ci serve conservarli,
    // ci servono solo i campionamenti prestazionali
    Matrix *mrBuffer = NULL;
    int curSampleIndex;
    int progress = 0, oldProgress = 0;
        
    /**
     * experiment[i, p] --> (m1[i], m2[i], products[p])
    */
    for (int i = 0; i < numM; i++){
        for (int p = 0; p < numProducts; p++){

            // to do an experiment, we must cycle through all its trials
            for (int t = 0; t < numTrials; t ++){
                
                mrBuffer = newMatrixEllpack();
                
                // initialize sample for the trial of this experiment
                curSampleIndex = i * numProducts * numTrials + p * numTrials + t;
                samples[curSampleIndex] = (Sample *)calloc(1, sizeof(Sample));
                samples[curSampleIndex] -> m1SampleId = msid1[i];
                samples[curSampleIndex] -> m2SampleId = msid2[i];
                samples[curSampleIndex] -> trial = t;

                // do the trial of this experiment and calculate its perfomance.
                ON_ERROR_LOG_AND_RETURN(products[p](m1[i], m2[i], mrBuffer, samples[curSampleIndex]), -1, "Error while doing trial %d for experiment %d, %d, %d\n", t, i, i, p);
                calcGflops(samples[curSampleIndex]);
                calcBandwidth(samples[curSampleIndex]);

                progress = (curSampleIndex * 100 / (numM * numProducts * numTrials));
                if (progress != oldProgress)
                {
                    oldProgress = progress;
                    logMsg(LOG_TAG_I, "Generated %d percent of samples\n", progress);
                }

                freeMatrixEllpack(mrBuffer);                    
            }
        }
    }

    return 0;
}

int printSamplesToCSV(int numSamples, Sample *samples[], char *filename){

    FILE *csv;
    csv = fopen(filename,"wb");

    //Stampo header
    fprintf(csv,"execTimeNsecs,");
    fprintf(csv,"productName,");
    fprintf(csv,"gflops,");
    fprintf(csv,"bandwidth,");
    fprintf(csv,"numElements_mat1,");
    fprintf(csv,"numBytes_mat1,");
    fprintf(csv,"name_mat1,");
    fprintf(csv, "format_mat1,");
    fprintf(csv,"numElements_mat2,");
    fprintf(csv,"numBytes_mat2,");
    fprintf(csv,"name_mat2\n");
    
    //Stampo un sample per ogni riga
    for(int i=0; i< numSamples; i++){

        fprintf(csv,"%ld,",samples[i]->execTimeNsecs + (samples[i]->execTimeSecs * 1e9));
        fprintf(csv,"%s,",samples[i]->productName);
        fprintf(csv,"%lf,",samples[i]->gflops);
        fprintf(csv,"%f,",samples[i]->bandwidth);
        fprintf(csv,"%ld,",samples[i]->m1SampleId->numElements);
        fprintf(csv,"%ld,",samples[i]->m1SampleId->numBytes);
        fprintf(csv,"%s,",samples[i]->m1SampleId->name);
        fprintf(csv,"%s,",samples[i]->m1SampleId->formatName);
        fprintf(csv,"%ld,",samples[i]->m2SampleId->numElements);
        fprintf(csv,"%ld,",samples[i]->m2SampleId->numBytes);
        fprintf(csv,"%s\n",samples[i]->m2SampleId->name);


    }

    fclose(csv);

    return 0;

}

/*************************************** MAIN ****************************************************/

int main(int argc, char *argv[]){
    
    const int NUM_M = NUM_MATRIX_FILE_NAMES * NUM_MATRIX_FORMATS * NUM_MV_WIDTHS;
    const int NUM_EXPERIMENTS = NUM_M * NUM_PRODUCTS * TRIALS;
    Matrix *mmMatrix, *mBuffer, *currentFormat, *currentMv;
    Matrix *m1s[NUM_M];
    Matrix *m2s[NUM_M];
    MatrixSampleID *m1sids[NUM_M], *m2sids[NUM_M], *currentM1sid, *currentM2sid;
    Sample *samples[NUM_EXPERIMENTS];
    int current;
    
    // set seed for random number generation
    SelectStream(0);
    PutSeed(SEED);
    logMsg(LOG_TAG_I, "Random number generator initialized\n");

    for (int i = 0; i < NUM_MATRIX_FILE_NAMES; i++){
            
        mmMatrix = newMatrixMM(MATRIX_FILE_NAMES[i]);
        
        // per ogni formato desiderato, converte la matrice letta in quel formato
        for (int f = 0; f < NUM_MATRIX_FORMATS; f ++){
            
            currentFormat = (Matrix *)MATRIX_FORMATS[f];
            mBuffer = currentFormat ->cloneEmpty(currentFormat);
            
            convertFromMM((DataMM*)mmMatrix ->data, mBuffer);

            logMsg(LOG_TAG_I, "Fine conversione %s\n",MATRIX_FILE_NAMES[i]);

            // costruisce un corrispondente oggetto MatrixSampleID
            currentM1sid = newMatrixSampleID(
                mBuffer ->numNonZero,
                mBuffer ->getSize(mBuffer),
                MATRIX_FILE_NAMES[i],
                currentFormat ->formatName
            );

            // crea un multivettore a partire dalla matrice sparsa per ogni
            // valore di width disponibile
            for(int w = 0; w < NUM_MV_WIDTHS; w++){
                
                current = i * NUM_MATRIX_FORMATS * NUM_MV_WIDTHS + f * NUM_MV_WIDTHS + w;
                currentMv = craftUniformMultiVectorFromMatrix(mBuffer, MV_WIDTHS[w], MV_MAX_VAL, MV_MIN_VAL);
                
                // costruisce un oggetto MatrixSampleID per il mv
                currentM2sid = newMatrixSampleID(
                     MV_WIDTHS[w],
                    currentMv ->getSize(currentMv),
                    (char *)calloc(128, sizeof(char)),
                    currentMv ->formatName
                    );
                snprintf((char *)currentM2sid ->name, 128, "k=%d", MV_WIDTHS[w]);
                
                // registra i riferimenti alle matrici, al multivettore e ai msids
                m1s[current] = mBuffer;
                m1sids[current] = currentM1sid;
                m2s[current] = currentMv;
                m2sids[current] = currentM2sid;
            }
        }
        freeMatrixMM(mmMatrix);
    }

    logMsg(LOG_TAG_I, "Matrices setup completed.\n");

    /**
     * moltiplica le matrici sparse con i corrispettivi multivettori
     * una volta per ogni funzione di prodotto matriciale desiderata
     * */
    doExperiments(m1s, m1sids, m2s, m2sids, NUM_M, PRODUCTS, NUM_PRODUCTS, TRIALS, samples);
    logMsg(LOG_TAG_I, "Experiments completed.\n");

    /**
     * scrive i risultati su file csv
     */
    char filename[128];
    snprintf(filename, 128, "results_%d.csv", SEED);
    printSamplesToCSV(NUM_EXPERIMENTS, samples, filename);
    logMsg(LOG_TAG_I, "Samples written to file %s.\n", filename);

    return 0;
}